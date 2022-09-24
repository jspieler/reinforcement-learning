"""Provides plots."""
import matplotlib.pyplot as plt 


def plot_avg_reward(avg_reward_list, fname) -> None:
    """Plots the average reward over the number of episodes.
    
    Args:
        avg_reward_list: A list containing the average rewards.
        fname: The name of the file to save the plot.
    """
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average episodic reward")
    plt.savefig(fname)