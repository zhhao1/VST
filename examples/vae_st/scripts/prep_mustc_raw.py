#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import torch.nn.functional as F
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.vae_st.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from fairseq import checkpoint_utils, utils, tasks
from fairseq.models.wav2vec.wav2vec2_laser import Wav2VecLaser
from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "utt_pre"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["dev","train", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)

        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)  
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)  # n_frames = waveform.size(1)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id, wav_path, offset

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
               
    for lang in args.languages:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        utt_rep_root = cur_root / "encoder_max_scale"
        utt_rep_root.mkdir(exist_ok=True)
        
        # Generate TSV manifest
        print("Generating manifest...")             
        train_text = []

        zip_path = cur_root / f"{utt_rep_root.name}.zip" 
        if not os.path.exists(zip_path):
            print("Generating zip...") 
            create_zip(utt_rep_root, zip_path)

        print("Fetching pretrain utt path...")
        utt_paths, utt_lengths = get_zip_manifest(zip_path)
   
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for waveform, sr, src_utt, tgt_utt, spk_id, utt_id, wav_path, offset in tqdm(dataset):
                length = int(waveform.size(1))
                manifest["id"].append(utt_id)
                manifest["audio"].append(f"{wav_path}:{offset}:{length}")
                manifest["n_frames"].append(length)
                manifest["tgt_text"].append(
                    src_utt if args.task == "asr" else tgt_utt
                )
                manifest["speaker"].append(spk_id)
                manifest["utt_pre"].append(utt_paths[utt_id] if utt_id in utt_paths.keys() else None)
                
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=1600, max_n_frames=480000)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy=None,
            extra={"use_audio_input": True}
            )
        # Clean up
        #shutil.rmtree(utt_rep_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--languages", type=str, nargs='+')
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
