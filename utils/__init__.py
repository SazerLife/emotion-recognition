from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def show_distribution(csv: pd.DataFrame, field_name: str, show: bool = True):
    assert field_name in csv.columns
    assert csv[field_name].dtypes in {np.dtype("float32"), np.dtype("float64")}

    fig, axis = plt.subplots(1, 2)
    fig.set_figwidth(15)
    fig.set_figheight(7)

    plt.title(field_name)
    hist = axis[0].hist(csv[field_name].to_numpy())

    plt.title(field_name)
    boxplot = axis[1].boxplot(csv[field_name].to_numpy())
    axis[1].grid(True)

    if not show:
        plt.close()

    return boxplot, hist


def make_audio_path(
    yyid: str,
    start_time: float,
    end_time: float,
    audio_dir=Path("data/CMU-MOSEI/Audio_chunk/Train_modified"),
):
    start_time = str(round(start_time, 4))
    end_time = str(round(end_time, 4))

    if len(start_time.split(".")[-1]) < 4:
        start_time += "0" * (4 - len(start_time.split(".")[-1]))
    if len(end_time.split(".")[-1]) < 4:
        end_time += "0" * (4 - len(end_time.split(".")[-1]))

    audio_path = audio_dir / f"{yyid}_{start_time}_{end_time}.wav"
    return audio_path


def get_all_audios_paths(
    csv: pd.DataFrame, audio_dir=Path("data/CMU-MOSEI/Audio_chunk/Train_modified")
):
    all_audios_paths = list()
    for row in tqdm(csv[["video", "start_time", "end_time"]].values):
        all_audios_paths.append(make_audio_path(*row, audio_dir=audio_dir))
    return all_audios_paths
