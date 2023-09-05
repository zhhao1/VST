import csv, argparse, os
import pandas as pd
from pathlib import Path
from examples.speech_to_text.data_utils import (
    save_df_to_tsv,
)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker", "tgt_text", "utt_pre"]

def load_samples_from_tsv(root: str, split: str):
    tsv_path = Path(root) / f"{split}.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    if len(samples) == 0:
        raise ValueError(f"Empty manifest: {tsv_path}")
    return samples

root = '/home/zhhao/data_source/MUST-C/en-de'
tsv_file = 'tst-COMMON_st'

samples = load_samples_from_tsv(root, tsv_file)

manifest0_5 = {c: [] for c in MANIFEST_COLUMNS}
manifest5_10 = {c: [] for c in MANIFEST_COLUMNS}
manifest10_15 = {c: [] for c in MANIFEST_COLUMNS}
manifest15_20 = {c: [] for c in MANIFEST_COLUMNS}
manifest20_50 = {c: [] for c in MANIFEST_COLUMNS}

for sample in samples:
    duration = int(sample['n_frames'])/16000 
    if duration < 5:
        manifest0_5["id"].append(sample["id"])
        manifest0_5["audio"].append(sample["audio"])
        manifest0_5["tgt_text"].append(sample["tgt_text"])
        manifest0_5["n_frames"].append(sample["n_frames"])
        manifest0_5["speaker"].append(sample["speaker"])
        manifest0_5["utt_pre"].append(sample["utt_pre"])
    elif duration < 10:
        manifest5_10["id"].append(sample["id"])
        manifest5_10["audio"].append(sample["audio"])
        manifest5_10["tgt_text"].append(sample["tgt_text"])
        manifest5_10["n_frames"].append(sample["n_frames"])
        manifest5_10["speaker"].append(sample["speaker"])
        manifest5_10["utt_pre"].append(sample["utt_pre"])         
    elif duration < 15:
        manifest10_15["id"].append(sample["id"])
        manifest10_15["audio"].append(sample["audio"])
        manifest10_15["tgt_text"].append(sample["tgt_text"])
        manifest10_15["n_frames"].append(sample["n_frames"])
        manifest10_15["speaker"].append(sample["speaker"])
        manifest10_15["utt_pre"].append(sample["utt_pre"]) 
    elif duration < 20:
        manifest15_20["id"].append(sample["id"])
        manifest15_20["audio"].append(sample["audio"])
        manifest15_20["tgt_text"].append(sample["tgt_text"])
        manifest15_20["n_frames"].append(sample["n_frames"])
        manifest15_20["speaker"].append(sample["speaker"])
        manifest15_20["utt_pre"].append(sample["utt_pre"]) 
    else:
        manifest20_50["id"].append(sample["id"])
        manifest20_50["audio"].append(sample["audio"])
        manifest20_50["tgt_text"].append(sample["tgt_text"])
        manifest20_50["n_frames"].append(sample["n_frames"])
        manifest20_50["speaker"].append(sample["speaker"])
        manifest20_50["utt_pre"].append(sample["utt_pre"])         
                    
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest0_5), os.path.join(root, tsv_file + '0_5.tsv')
)
            
save_df_to_tsv(
    pd.DataFrame.from_dict(manifest5_10), os.path.join(root, tsv_file + '5_10.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest10_15), os.path.join(root, tsv_file + '10_15.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest15_20), os.path.join(root, tsv_file + '15_20.tsv')
)

save_df_to_tsv(
    pd.DataFrame.from_dict(manifest20_50), os.path.join(root, tsv_file + '20_50.tsv')
)
'''
	base	vae_st
0_5: 	23.98	24.86
5_10:	24.41	25.26
10_15:	23.50	23.79
15_20:	22.82	22.79
20_50:	19.09	22.25
all:	23.86	24.51
'''

