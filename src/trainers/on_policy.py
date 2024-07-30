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

    avg_total_reward = [0.0 for _ in range(env.max_num_agents)]
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

        if not Logger:
            return
        for agent_idx, r in enumerate(total_reward):
            avg_total_reward[agent_idx] += r
            if episode % 50 == 0:
                agent = agents[agent_idx]
                logger.report_scalar(
                    title="Reward",
                    series=agent.__repr__(),
                    value=avg_total_reward[agent_idx] / 50,
                    iteration=episode,
                )

                avg_total_reward[agent_idx] = 0.0
