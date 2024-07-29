import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from envs import Env
from agents import Agent
from tqdm import tqdm


def test(env: Env, agents: list[Agent], episodes: int):
    for agent in agents:
        agent.TRAIN = False

    # Create agents map
    agents_map = {}
    for idx, agent in enumerate(env.possible_agents):
        agents_map[agent] = idx

    rewards = np.zeros([env.max_num_agents, episodes], dtype=np.float32)
    for episode in tqdm(range(episodes)):
        env.reset()

        for agent_idx in env.agent_iter():
            agent = agents[agents_map[agent_idx]]
            obs, r, terminate, truncate, info = env.last()
            rewards[agents_map[agent_idx], episode] += r

            assert obs
            if not (terminate or truncate):
                action = agent.get_action(obs)
            else:
                action = None

            env.step(action)

    return rewards


def plot_reward_histograms(rewards: np.ndarray, titles=None):
    [agents, episodes] = rewards.shape
    fig, axes = plt.subplots(1, agents, sharex=True, sharey=True)

    for idx in range(agents):
        ax = sns.histplot(rewards[idx], stat="percent", ax=axes[idx])
        if idx == 0:
            ax.set_ylabel("Percent")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Reward")
        if titles:
            ax.set_title(titles[idx])

    plt.show()
