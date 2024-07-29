from tqdm import tqdm
from clearml import Logger

from agents import Agent
from envs import Env
from utils import rotate_list
from typing import Callable


def train_on_policy(
    env: Env,
    agents: list[Agent],
    episodes,
    on_episode_complete: Callable = lambda ep: None,
):
    logger = Logger.current_logger()

    # Create agents map
    agents_map = {}
    for idx, agent in enumerate(env.possible_agents):
        agents_map[agent] = idx

    for episode in tqdm(range(1, episodes + 1)):
        env.reset()
        total_reward = [0.0 for i in range(env.num_agents)]

        for agent_idx in env.agent_iter():
            agent = agents[agents_map[agent_idx]]
            obs, r, terminate, truncate, info = env.last()
            total_reward[agents_map[agent_idx]] += r

            assert obs
            agent.observe(obs, r, terminate, truncate, info)
            if not (terminate or truncate):
                action = agent.get_action(obs)
            else:
                action = None

            agent.update(obs, r, action, terminate, truncate, info)
            env.step(action)

        # callback function
        on_episode_complete(episode)

        for agent_idx, r in enumerate(total_reward):
            agent = agents[agent_idx]
            if logger:
                logger.report_scalar(
                    title="Reward", series=agent.__repr__(), value=r, iteration=episode
                )
