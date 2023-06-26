import argparse
import json
from pathlib import Path
from typing import Any, Dict

import nltk
import pandas as pd
from gensim.models import FastText
from tqdm import tqdm

from utils.parameters import PUNCTUATION, SEED


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", type=Path)
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
    config_path: Path = args.config
    config: Dict[str, Dict[str, Any]] = json.loads(config_path.read_text())

    train_path = config["data"]["train"]
    text_column = config["data"]["text_column"]
    model_params = config["model"]
    epochs = config["train"]["epochs"]
    save_dir = config_path.parent / "checkpoint"

    train_mod_csv = pd.read_csv(train_path)
    X_train = [tokenize(text) for text in tqdm(train_mod_csv[text_column].to_list())]

    fasttext = FastText(X_train, **model_params)
    fasttext.build_vocab(corpus_iterable=X_train)
    fasttext.train(X_train, total_examples=len(X_train), epochs=epochs)

    save_dir.mkdir(parents=True, exist_ok=True)
    fasttext.wv.save_word2vec_format(str(save_dir / "keyed_vectors.txt"), binary=False)
    fasttext.save(str(save_dir / "fasttext.bin"))

    # selfmade_fasttext = gensim.models.KeyedVectors.load_word2vec_format(
    #     "models/self-made/fasttext/keyed_vectors.txt",
    #     unicode_errors="ignore",
    #     binary=False,
    # )
