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

    utt_model = Wav2VecLaser.from_pretrained(args.pretrain_utt_path, checkpoint_file=args.pretrain_utt_name).models[0]
    utt_model.cuda()
            
    for lang in args.languages:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        utt_rep_root = cur_root / "encoder_max_scale"
        os.makedirs(utt_rep_root, exist_ok=True)
        
        i = 0
        # Generate TSV manifest
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            dataset = MUSTC(args.data_root, lang, split)
            for waveform, sr, src_utt, tgt_utt, spk_id, utt_id, wav_path, offset in tqdm(dataset):
                length = int(waveform.size(1))
                if i >= args.process_number * (args.multi-1) and i < args.process_number * args.multi:
                    if length > 480000 and split == 'train':
                        pass                    
                    elif not os.path.exists(utt_rep_root / f"{utt_id}.npy"):
                        with torch.no_grad():
                            waveform = waveform.float().squeeze(0)
                            waveform = F.layer_norm(waveform, waveform.shape).unsqueeze(0).cuda()
                            padding_mask = torch.Tensor([False]*waveform.shape[1]).cuda()
                            sample = {'padding_mask': padding_mask, 'source': waveform}
                            embedding = utt_model(**sample)
                            np.save(utt_rep_root / f"{utt_id}.npy", embedding.squeeze(0).cpu())
                    else:
                        pass
                i = i + 1
                                                            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--languages", type=str, nargs='+')
    parser.add_argument("--pretrain-utt-path", type=str, help="path to pretrained sentence level model")
    parser.add_argument("--pretrain-utt-name", type=str, help="name of pretrained sentence level model")
    parser.add_argument("--process-number", type=int)
    parser.add_argument("--multi", type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
