"""
Examples:
python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp1-markuped-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Train_modified.csv \
    --text-column text \
    --target-dir data/CMU-MOSEI/fasttext_featrues_markuped_text/train_modified

python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp1-markuped-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Val_modified.csv \
    --text-column text \
    --target-dir data/CMU-MOSEI/fasttext_featrues_markuped_text/val_modified

python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp1-markuped-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Test_original_without_neg_time.csv \
    --text-column text \
    --target-dir data/CMU-MOSEI/fasttext_featrues_markuped_text/test_original

    
python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp2-recognised-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Train_modified.csv \
    --text-column ASR \
    --target-dir data/CMU-MOSEI/fasttext_featrues_recognised_text/train_modified

python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp2-recognised-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Val_modified.csv \
    --text-column ASR \
    --target-dir data/CMU-MOSEI/fasttext_featrues_recognised_text/val_modified

python fasttext_extract_features.py \
    --model-path experiments/fasttext/exp2-recognised-text/checkpoint/keyed_vectors.txt \
    --csv-path data/CMU-MOSEI/Labels/Data_Test_original_without_neg_time.csv \
    --text-column ASR \
    --target-dir data/CMU-MOSEI/fasttext_featrues_recognised_text/test_original
"""
import argparse
from pathlib import Path

import gensim
import nltk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils.parameters import PUNCTUATION


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--text-column", type=str)
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

    model_path: Path = args.model_path
    csv_path: Path = args.csv_path
    text_column: str = args.text_column
    target_dir: Path = args.target_dir

    assert model_path.exists() and csv_path.exists()
    target_dir.mkdir(parents=True, exist_ok=True)

    fasttext = gensim.models.KeyedVectors.load_word2vec_format(
        model_path, unicode_errors="ignore", binary=False
    )

    csv = pd.read_csv(csv_path)[["video", "start_time", "end_time", text_column]]

    for ytid, start_time, end_time, text in tqdm(csv.values):
        start_time = float(start_time)
        end_time = float(end_time)
        file_name = f"{ytid}_{start_time:.4f}_{end_time:.4f}.npy"

        try:
            vector = fasttext.get_mean_vector(tokenize(text))
            np.save(target_dir / file_name, vector)
        except Exception as exception:
            print()
            print(text)
            print(tokenize(text))
            print(exception)
