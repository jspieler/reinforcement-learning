"""Provides plots."""
from typing import List

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_avg_reward(avg_reward_list: List[float], fname: str) -> None:
    """Plots the average reward over the number of episodes.

    Args:
        avg_reward_list: A list containing the average rewards.
        fname: The name of the file to save the plot.
    """
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average episodic reward")
    plt.savefig(fname)


def plot_rewards(
    data: List[pd.DataFrame],
    xaxis: str = "Episode",
    value: str = "EpisodicReward",
    smooth: int = 1,
) -> None:
    """Plots the rewards of multiple algorithms."""
    # smooth data with moving window average
    if smooth > 1:
        for df in data:
            df[value] = df[value].rolling(smooth, min_periods=1).mean()

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)
    g = sns.lineplot(data=data, x=xaxis, y=value, hue="algorithm", ci="sd")
    plt.legend(
        loc="upper center",
        ncol=3,
        handlelength=1,
        mode="expand",
        borderaxespad=0.0,
        prop={"size": 13},
    )
    # when hue is used seaborn assigns it as a legend title
    g.legend_.set_title(None)
    g.legend_.texts[0].set_text("")

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout(pad=0.5)
