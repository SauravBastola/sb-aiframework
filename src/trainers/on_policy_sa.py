from tqdm import tqdm
from clearml import Logger

from agents import Agent
from utils import rotate_list
from typing import Callable

from gymnasium import Env


def train_on_policy_sa(
    env: Env,
    agent,
    episodes,
    on_episode_complete: Callable = lambda ep: None,
):
    logger = Logger.current_logger()

    avg_total_reward = 0.0
    for episode in tqdm(range(1, episodes + 1)):
        obs, info = env.reset()

        episode_sample = []
        done = False
        while not done:
            action = agent.get_action(obs, [_ for _ in range(env.action_space.n)])
            next_obs, r, terminate, truncate, info = env.step(action)
            done = terminate or truncate
            episode_sample.append((obs, action, r, next_obs, done))
            obs = next_obs

        total_reward = 0.0
        for idx, sample in enumerate(episode_sample):
            obs, action, r, obs_next, done = sample
            action_next = None
            if not done:
                action_next = episode_sample[idx + 1][1]

            total_reward += r

            agent.update(obs, action, r, obs_next, action_next, done)

        # callback function
        on_episode_complete(episode)

        if not logger:
            continue

        avg_total_reward += total_reward
        if episode % 5 == 0:
            logger.report_scalar(
                title="Reward",
                series=agent.__repr__(),
                value=avg_total_reward / 5,
                iteration=episode,
            )

            avg_total_reward = 0.0
