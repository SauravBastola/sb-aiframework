from tqdm import tqdm
from clearml import Logger

from agents import Agent
from utils import rotate_list
from typing import Callable

from gymnasium import Env


def train_pgm_sa(
    env: Env,
    agent,
    episodes,
    on_episode_complete: Callable = lambda ep: None,
):
    logger = Logger.current_logger()

    avg_total_reward = 0.0
    for episode in tqdm(range(1, episodes + 1)):
        saved_log_probs = []
        rewards = []

        obs, info = env.reset()

        episode_sample = []
        done = False

        total_reward = 0.0
        while not done:
            action, log_prob = agent.get_action(obs)
            saved_log_probs.append(log_prob)
            next_obs, r, terminate, truncate, info = env.step(action)
            done = terminate or truncate
            rewards.append(r)
            obs = next_obs

            total_reward += r

        agent.update(saved_log_probs, rewards)

        # callback function
        on_episode_complete(episode)

        if not logger:
            continue

        avg_total_reward += total_reward
        if episode % 10 == 0:
            logger.report_scalar(
                title="Reward",
                series=agent.__repr__(),
                value=avg_total_reward / 10,
                iteration=episode,
            )

            avg_total_reward = 0.0
