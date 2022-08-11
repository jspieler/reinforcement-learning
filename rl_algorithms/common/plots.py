import matplotlib.pyplot as plt 


def plot_avg_reward(avg_reward_list, fname):
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average episodic reward")
    plt.savefig(fname)