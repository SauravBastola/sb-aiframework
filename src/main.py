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

from agents import DQNAgentSA, DQNAgentMA, RandomAgentSA, RandomAgentMA
from envs import ProbeEnv1, BhoosGameEnv
from trainers import train_on_policy_sa, train_on_policy_ma
from test_sa import test_sa, plot_reward_histograms_sa
from test_ma import test_ma, plot_reward_histograms_ma


app = typer.Typer()


def create_agent(agent, env: BhoosGameEnv, config: Dict):
    variant = env.game_variant

    assert isinstance(variant.action_space, gymnasium.spaces.Discrete)
    assert isinstance(variant.observation_space, gymnasium.spaces.Discrete)

    if agent == "DQN":
        return DQNAgentMA(
            variant.observation_space.n,
            variant.action_space.n,
            config["hidden_layer_size"],
        )
    else:
        raise Exception("Invalid agent name")


# ------------------------- PettingZoo Environments -------------------------


class PettingZooEnvs(str, Enum):
    t3 = "t3"
    texas_holem = "texas-holdem"


def make_petting_zoo_env(env_name):
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

    return env, obs_space, action_space


@app.command("pztrain")
def petting_zoo(
    env_name: PettingZooEnvs,
    eps: int = 10_000,
    log: bool = False,
    agent: str = "DQN",
):
    env, obs_space, action_space = make_petting_zoo_env(env_name)

    config = {
        "num_players": 2,
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    random_id = "".join(random.choices(string.ascii_lowercase, k=8))

    task_name = env_name + "-" + random_id
    if log:
        task: Task | None = Task.init(
            project_name="petting-zoo-envs",
            task_name=task_name,
            reuse_last_task_id=False,
        )
        task.connect_configuration(config, "GameInfo")
    else:
        task = None

    lagent = DQNAgentMA(obs_space, action_space, config["hidden_layer_size"])
    ragents = [RandomAgentMA("r" + str(_)) for _ in range(env.max_num_agents - 1)]
    agents = [*ragents, lagent]

    # callback function for saving the model
    def save_model(episode):
        if episode % 200_000 == 0:
            lagent.save_model(f"../models/{task_name}-{episode}")

    train_on_policy_ma(env, agents, config["episodes"], on_episode_complete=save_model)

    # Test and log, plot results
    test_episodes = 1000
    rewards = test_ma(env, agents, test_episodes)
    for idx, agent in enumerate(agents):
        title = "Avg. Test Reward " + agent.__repr__()
        avg = np.sum(rewards[idx]) / test_episodes
        print(title, avg)
        if task:
            task.get_logger().report_single_value(title, avg)

    plot_reward_histograms_ma(rewards, [agent.__repr__() for agent in agents])

    lagent.save_model(f"../models/{task_name}")


@app.command("pztest")
def petting_zoo_test(
    env_name: PettingZooEnvs,
    eps: int = 10_000,
    agent: str = "DQN",
):
    env, obs_space, action_space = make_petting_zoo_env(env_name)

    config = {
        "num_players": 2,
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    # lagent = DQNAgentMA(obs_space, action_space, config["hidden_layer_size"])
    lagent = RandomAgentMA(id="l")
    ragents = [RandomAgentMA(id="r" + str(_)) for _ in range(env.max_num_agents - 1)]
    agents = [lagent, *ragents]

    # Test and log, plot results
    test_episodes = eps
    rewards = test_ma(env, agents, test_episodes)

    for idx, agent in enumerate(agents):
        avg = np.sum(rewards[idx]) / test_episodes
        print("Avg. Test Reward " + agent.__repr__(), avg)

    plot_reward_histograms_ma(rewards, [agent.__repr__() for agent in agents])


@app.command("probe1")
def probe1_command():
    lagent = DQNAgent(1, 1, 4)
    lagent.update_batch_size = 1
    p1 = ProbeEnv1()
    train_on_policy_ma(p1, [lagent], 2000)

    value = lagent._value([0], 0).item()
    ok = abs(value - 1) < 0.01

    print(f"Expected value = 1 Learned = {value}")
    print("OK" if ok else "Not OK")

    if not ok:
        exit(1)


# -------------------------    GYM Environments    -------------------------


class GymEnvs(str, Enum):
    cartpole = "cartpole"
    mountain_car = "mountain-car"
    acrobot = "acrobot"
    lunar_lander = "lunar-lander"
    blackjack = "blackjack"
    taxi = "taxi"
    cliff_walking = "cliff-walking"
    frozen_lake = "frozen-lake"


def make_gym_env(env_name):
    import gymnasium as gym

    if env_name == "cartpole":
        env = gym.make("CartPole-v1")
        obs_space = 4
        action_space = 2
        state_shape = None
    elif env_name == "mountain-car":
        env = gym.make("MountainCar-v0")
        obs_space = 2
        action_space = 3
        state_shape = None
    elif env_name == "acrobot":
        env = gym.make("Acrobot-v1")
        obs_space = 6
        action_space = 3
        state_shape = None
    elif env_name == "lunar-lander":
        env = gym.make("LunarLander-v2")
        obs_space = 8
        action_space = 4
        state_shape = None
    elif env_name == "blackjack":
        env = gym.make("Blackjack-v1")
        obs_space = 32 + 11 + 2
        action_space = 2
        state_shape = (32, 11, 2)
    elif env_name == "taxi":
        env = gym.make("Taxi-v3")
        obs_space = 500
        action_space = 6
        state_shape = [500]
    elif env_name == "cliff-walking":
        env = gym.make("CliffWalking-v0")
        obs_space = 48
        action_space = 4
        state_shape = None
    elif env_name == "frozen-lake":
        env = gym.make("FrozenLake-v1")
        obs_space = 16
        action_space = 4
        state_shape = None
    else:
        print("Invalid Game")
        sys.exit(1)
    return env, obs_space, action_space, state_shape


@app.command("gymtrain")
def gymnasium_env(
    env_name: GymEnvs,
    eps: int = 500,
    log: bool = False,
    agent: str = "DQN",
):
    env, obs_space, action_space, state_shape = make_gym_env(env_name)

    config = {
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    random_id = "".join(random.choices(string.ascii_lowercase, k=8))

    task_name = env_name + "-" + random_id
    if log:
        task: Task | None = Task.init(
            project_name="gymnasium-envs",
            task_name=task_name,
            reuse_last_task_id=False,
        )
        task.connect_configuration(config, "GameInfo")
    else:
        task = None

    lagent = DQNAgentSA(
        obs_space, action_space, config["hidden_layer_size"], state_shape=state_shape
    )

    # callback function for saving the model
    def save_model(episode):
        if episode % 200_000 == 0:
            lagent.save_model(f"../models/{task_name}-{episode}")

    train_on_policy_sa(env, lagent, config["episodes"], on_episode_complete=save_model)

    # Test and log, plot results
    test_episodes = 1000
    rewards = test_sa(env, lagent, test_episodes)
    env.close()
    title = "Avg. Test Reward " + lagent.__repr__()
    avg = np.sum(rewards) / test_episodes
    print(title, avg)
    if task:
        task.get_logger().report_single_value(title, avg)

    plot_reward_histograms_sa(rewards, lagent.__repr__())

    lagent.save_model(f"../models/{task_name}")


@app.command("gymtest")
def gymnasium_env_test(
    env_name: GymEnvs,
    eps: int = 10_000,
    agent: str = "DQN",
):
    env, obs_space, action_space, state_shape = make_gym_env(env_name)

    config = {
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    lagent = RandomAgentSA(id="gym")

    # Test and log, plot results
    test_episodes = eps
    rewards = test_sa(env, lagent, test_episodes)
    env.close()

    avg = np.sum(rewards) / test_episodes
    print("Avg. Test Reward " + lagent.__repr__(), avg)

    plot_reward_histograms_sa(rewards, lagent.__repr__())


@app.command("gymtest-viz")
def gymnasium_env_test_viz(
    env_name: GymEnvs,
    eps: int = 10,
    agent: str = "DQN",
):
    import gymnasium as gym

    env, obs_space, action_space, state_shape = make_gym_env(env_name)

    config = {
        "hidden_layer_size": 64,
        "episodes": eps,
    }

    lagent = RandomAgentSA(id="gym")

    # Test and log, plot results
    test_episodes = eps
    rewards = test_sa(env, lagent, test_episodes)
    env.close()

    avg = np.sum(rewards) / test_episodes
    print("Avg. Test Reward " + lagent.__repr__(), avg)


if __name__ == "__main__":
    app()
