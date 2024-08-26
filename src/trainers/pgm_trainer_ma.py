from tqdm import tqdm
from clearml import Logger

from agents import Agent
from utils import rotate_list
from typing import Callable

from envs import Env


def train_pgm_ma(
    env: Env,
    agents,
    episodes,
    on_episode_complete: Callable = lambda ep: None,
):
    logger = Logger.current_logger()

    agents_map = {}
    for idx, agent in enumerate(env.possible_agents):
        agents_map[agent] = idx

    avg_total_reward = [0.0 for _ in range(env.max_num_agents)]
    for episode in tqdm(range(1, episodes + 1)):
        rotate_list(agents)
        env.reset()

        total_reward = [0.0 for i in range(env.num_agents)]

        saved_log_probs = []
        rewards = []

        for agent_idx in env.agent_iter():
            agent = agents[agents_map[agent_idx]]
            obs, r, terminate, truncate, info = env.last()
            total_reward[agents_map[agent_idx]] += r

            assert obs
            agent.observe(obs, r, terminate, truncate, info)
            if not (terminate or truncate):
                action, log_prob = agent.get_action(obs)
            else:
                action, log_prob = None, None

            if agents_map[agent_idx] == 0:
                saved_log_probs.append(log_prob)
                rewards.append(r)

            env.step(action)

        for agent in agents:
            agent.update(saved_log_probs[:-1], rewards[1:])

        # callback function
        on_episode_complete(episode)

        if not logger:
            continue

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
