"""
Examples:
python mel_mfcc_extract_features.py \
    --csv-path data/CMU-MOSEI/Labels/Data_Train_modified.csv \
    --source-dir data/CMU-MOSEI/Audio_chunk/Train_modified \
    --target-dir data/CMU-MOSEI/audio_featrues/train_modified

python mel_mfcc_extract_features.py \
    --csv-path data/CMU-MOSEI/Labels/Data_Val_modified.csv \
    --source-dir data/CMU-MOSEI/Audio_chunk/Val_modified \
    --target-dir data/CMU-MOSEI/audio_featrues/val_modified

python mel_mfcc_extract_features.py \
    --csv-path data/CMU-MOSEI/Labels/Data_Test_original_without_neg_time.csv \
    --source-dir data/CMU-MOSEI/Audio_chunk/Test_original \
    --target-dir data/CMU-MOSEI/audio_featrues/test_original
"""
import argparse
from pathlib import Path

import gensim
import nltk
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
import soundfile as sf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils.parameters import PUNCTUATION


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    return parser


def tokenize(text: str, remove_punctuation: bool = True) -> list[str]:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    if remove_punctuation:
        tokens = [token for token in tokens if token not in PUNCTUATION]
    return tokens


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    csv_path: Path = args.csv_path
    source_dir: Path = args.source_dir
    target_dir: Path = args.target_dir
    assert source_dir.exists() and csv_path.exists()
    target_dir.mkdir(parents=True, exist_ok=True)

    MelSpectrogram_transform = T.MelSpectrogram(
        16000,
        n_fft=480,
        hop_length=160,
        f_max=8000,
        center=True,
        n_mels=95,
        window_fn=torch.hamming_window,
        normalized=True,
        mel_scale="slaney",
    )
    MFCC_transform = T.MFCC(
        16000,
        n_mfcc=95,
        log_mels=True,
        melkwargs={
            "n_fft": 480,
            "hop_length": 160,
            "n_mels": 95,
            "center": True,
            "window_fn": torch.hamming_window,
            "normalized": True,
            "f_max": 8000,
        },
    )

    csv = pd.read_csv(csv_path)[["video", "start_time", "end_time"]]

    for ytid, start_time, end_time in tqdm(csv.values):
        stem = f"{ytid}_{float(start_time):.4f}_{float(end_time):.4f}"

        try:
            audio, sr = sf.read(source_dir / f"{stem}.wav", dtype=np.float32)
            audio = audio[0 : sr * 5]
            padded_audio = np.zeros(sr * 5, dtype=audio.dtype)
            padded_audio[0 : audio.shape[0]] = audio
            audio = torch.from_numpy(padded_audio).unsqueeze(0)

            mel_spectrogram = MelSpectrogram_transform(audio)
            mfcc = MFCC_transform(audio)
            feature = torch.cat((mel_spectrogram, mfcc), dim=0)
            torch.save(feature, target_dir / f"{stem}.pt")

        except Exception as exception:
            print(exception)
