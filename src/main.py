from os import environ
import random
import string
from typing import Dict, Literal
from pettingzoo.utils.env import gymnasium
import sys
from clearml import Task
import numpy as np

import typer
from enum import Enum

from agents import (
    DQNAgent,
    RandomAgent,
    GymAgent,
    BaselineBotAgent,
)
from envs import ProbeEnv1, BhoosGameEnv
from GameInfo import GameInfo
from trainers import train_on_policy
from test import test, plot_reward_histograms


app = typer.Typer()


def create_agent(agent, env: BhoosGameEnv, config: Dict):
    variant = env.game_variant

    assert isinstance(variant.action_space, gymnasium.spaces.Discrete)
    assert isinstance(variant.observation_space, gymnasium.spaces.Discrete)

    if agent == "DQN":
        return DQNAgent(
            variant.observation_space.n,
            variant.action_space.n,
            config["hidden_layer_size"],
        )
    else:
        raise Exception("Invalid agent name")


class PettingZooEnvs(str, Enum):
    t3 = "t3"
    texas_holem = "texas-holdem"


@app.command("pztrain")
def petting_zoo(
    env_name: PettingZooEnvs,
    eps: int = 10_000,
    log: bool = False,
    agent: str = "DQN",
):
    from pettingzoo.classic import texas_holdem_v4, tictactoe_v3

    # env = texas_holdem_v4.env(render_mode="human")
    if env_name == "texas-holdem":
        env = texas_holdem_v4.env()
        obs_space = 72
        action_space = 4
    elif env_name == "t3":
        env = tictactoe_v3.env()
        obs_space = 18
        action_space = 9
    else:
        print("Invalid Game")
        sys.exit(1)

    config = {
        "num_players": 2,
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    random_id = "".join(random.choices(string.ascii_lowercase, k=8))

    task_name = env_name + "-" + random_id
    if log:
        task: Task | None = Task.init(
            project_name="petting-zoo-" + env_name,
            task_name=task_name,
            reuse_last_task_id=False,
        )
        task.connect_configuration(config, "GameInfo")
    else:
        task = None

    lagent = DQNAgent(obs_space, action_space, config["hidden_layer_size"])
    ragents = [RandomAgent() for _ in range(env.max_num_agents - 1)]
    agents = [lagent, *ragents]

    # callback function for saving the model
    def save_model(episode):
        if episode % 200_000 == 0:
            lagent.save_model(f"../models/{task_name}-{episode}")

    for agent in agents:
        if isinstance(agent, GymAgent):
            agent.start()

    train_on_policy(env, agents, config["episodes"], on_episode_complete=save_model)

    for agent in agents:
        if isinstance(agent, GymAgent):
            agent.stop()

    # Test and log, plot results
    test_episodes = 1000
    rewards = test(env, agents, test_episodes)
    for idx, agent in enumerate(agents):
        title = "Avg. Test Reward " + agent.__repr__()
        avg = np.sum(rewards[idx]) / test_episodes
        print(title, avg)
        if task:
            task.get_logger().report_single_value(title, avg)

    plot_reward_histograms(rewards, [agent.__repr__() for agent in agents])

    lagent.save_model(f"../models/{task_name}")


@app.command("ptest")
def petting_zoo_test(
    env_name: PettingZooEnvs,
    eps: int = 10_000,
    agent: str = "DQN",
):
    from pettingzoo.classic import texas_holdem_v4, tictactoe_v3

    if env_name == "texas-holdem":
        env = texas_holdem_v4.env()
        obs_space = 72
        action_space = 4
    elif env_name == "t3":
        env = tictactoe_v3.env()
        obs_space = 18
        action_space = 9
    else:
        print("Invalid Game")
        sys.exit(1)

    config = {
        "num_players": 2,
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    lagent = DQNAgent(obs_space, action_space, config["hidden_layer_size"])
    ragents = [RandomAgent() for _ in range(env.max_num_agents - 1)]
    agents = [lagent, *ragents]

    # Test and log, plot results
    test_episodes = eps
    rewards = test(env, agents, test_episodes)

    for idx, agent in enumerate(agents):
        avg = np.sum(rewards[idx]) / test_episodes
        print("Avg. Test Reward " + agent.__repr__(), avg)

    plot_reward_histograms(rewards, [agent.__repr__() for agent in agents])


@app.command("probe1")
def probe1_command():
    lagent = DQNAgent(1, 1, 4)
    lagent.update_batch_size = 1
    p1 = ProbeEnv1()
    train_on_policy(p1, [lagent], 2000)

    value = lagent._value([0], 0).item()
    ok = abs(value - 1) < 0.01

    print(f"Expected value = 1 Learned = {value}")
    print("OK" if ok else "Not OK")

    if not ok:
        exit(1)


if __name__ == "__main__":
    app()
