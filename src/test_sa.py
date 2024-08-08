import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from envs import Env
from agents import Agent
from tqdm import tqdm


def test_sa(env, agent, episodes: int):
    agent.TRAIN = False

    rewards = np.zeros([episodes], dtype=np.float32)
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()

        done = False
        while not done:
            action, a = agent.get_action(obs, [_ for _ in range(env.action_space.n)])
            next_obs, r, terminate, truncate, info = env.step(action)
            done = terminate or truncate
            rewards[episode] += r
            obs = next_obs

    return rewards


def test_pgm_sa(env, agent, episodes: int):
    agent.TRAIN = False

    rewards = np.zeros([episodes], dtype=np.float32)
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()

        done = False
        while not done:
            action, a = agent.get_action(obs)
            next_obs, r, terminate, truncate, info = env.step(action)
            done = terminate or truncate
            rewards[episode] += r
            obs = next_obs

    return rewards


def plot_reward_histograms_sa(rewards: np.ndarray, title=None):

    fig, axis = plt.subplots(1, sharex=True, sharey=True)
    ax = sns.histplot(rewards, stat="percent", ax=axis)
    ax.set_ylabel("Percent")
    ax.set_xlabel("Reward")
    if title:
        ax.set_title(title)

    plt.show()
