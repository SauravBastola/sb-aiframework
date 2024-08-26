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
    DQNAgentSA,
    DQNAgentSA2,
    RandomAgentSA,
    PGNAgentSA,
    PGNBaselineAgentSA,
    DQNAgentMA,
    DQNAgentMA_Dummy,
    PGNAgentMA,
    PGNAgentMA_Dummy,
    RandomAgentMA,
)
from envs import ProbeEnv1, BhoosGameEnv

from trainers import (
    train_on_policy_sa,
    train_on_policy_ma,
    train_pgm_sa,
    train_pgm_ma,
    train_pgm_baseline_sa,
)
from test_sa import test_sa, test_pgm_sa, plot_reward_histograms_sa
from test_ma import test_ma, test_pgm_ma, plot_reward_histograms_ma

app = typer.Typer()


def create_agent_sa(agent, obs_space, action_space, state_shape, config: Dict):
    if agent == "DQN":
        return DQNAgentSA(
            obs_space, action_space, config["hidden_layer_size"], state_shape
        )

    elif agent == "DQN2":
        return DQNAgentSA2(
            obs_space, action_space, config["hidden_layer_size"], state_shape
        )

    elif agent == "PGN":
        return PGNAgentSA(
            obs_space, action_space, config["hidden_layer_size"], state_shape
        )

    elif agent == "PGNBaseline":
        return PGNBaselineAgentSA(
            obs_space, action_space, config["hidden_layer_size"], state_shape
        )

    elif agent == "Random":
        return RandomAgentSA(id="gym")
    else:
        raise Exception("Invalid agent name")


def create_trainer_sa(agent):
    trainer = train_on_policy_sa
    if agent == "PGN":
        trainer = train_pgm_sa
    elif agent == "PGNBaseline":
        trainer = train_pgm_baseline_sa
    return trainer


def create_tester_sa(agent):
    tester = test_sa
    if agent == "PGN":
        tester = test_pgm_sa
    elif agent == "PGNBaseline":
        tester = test_pgm_sa
    return tester


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


def make_gym_env(env_name, viz):
    import gymnasium as gym

    if viz:
        render_mode = "human"
    else:
        render_mode = None

    if env_name == "cartpole":
        env = gym.make("CartPole-v1", render_mode=render_mode)
        obs_space = 4
        action_space = 2
        state_shape = None
    elif env_name == "mountain-car":
        env = gym.make("MountainCar-v0", render_mode=render_mode)
        obs_space = 2
        action_space = 3
        state_shape = None
    elif env_name == "acrobot":
        env = gym.make("Acrobot-v1", render_mode=render_mode)
        obs_space = 6
        action_space = 3
        state_shape = None
    elif env_name == "lunar-lander":
        env = gym.make("LunarLander-v2", render_mode=render_mode)
        obs_space = 8
        action_space = 4
        state_shape = None
    elif env_name == "blackjack":
        env = gym.make("Blackjack-v1", render_mode=render_mode)
        obs_space = 32 + 11 + 2
        action_space = 2
        state_shape = (32, 11, 2)
    elif env_name == "taxi":
        env = gym.make("Taxi-v3", render_mode=render_mode)
        obs_space = 500
        action_space = 6
        state_shape = [500]
    elif env_name == "cliff-walking":
        env = gym.make("CliffWalking-v0", render_mode=render_mode)
        obs_space = 48
        action_space = 4
        state_shape = [48]
    elif env_name == "frozen-lake":
        env = gym.make("FrozenLake-v1", render_mode=render_mode)
        obs_space = 16
        action_space = 4
        state_shape = [16]
    else:
        print("Invalid Game")
        sys.exit(1)
    return env, obs_space, action_space, state_shape


@app.command("gymtrain")
def gymnasium_env(
    env_name: GymEnvs,
    eps: int = 500,
    log: bool = False,
    retrain: bool = False,
    model: str = "",
    agent: str = "DQN",
    hidden_size: int = 64,
):
    model_path = model
    env, obs_space, action_space, state_shape = make_gym_env(env_name, False)

    config = {"hidden_layer_size": hidden_size, "episodes": eps, "agent": agent}
    if retrain:
        config["base_model"] = model_path.split("/")[-1]

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

    lagent = create_agent_sa(agent, obs_space, action_space, state_shape, config)
    if retrain:
        lagent.load_model(model_path)

    # callback function for saving the model
    def save_model(episode):
        if episode % 200_000 == 0:
            lagent.save_model(f"../models/{task_name}-{episode}")

    train = create_trainer_sa(agent)
    train(env, lagent, config["episodes"], on_episode_complete=save_model)

    # Test and log, plot results
    test_episodes = 1000
    test = create_tester_sa(agent)
    rewards = test(env, lagent, test_episodes)
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
    eps: int = 1_000,
    agent: str = "DQN",
    model: str = "",
    viz: bool = False,
    hidden_size: int = 64,
):
    model_path = model
    env, obs_space, action_space, state_shape = make_gym_env(env_name, viz)

    config = {
        "hidden_layer_size": hidden_size,
        "episodes": eps,
    }

    lagent = create_agent_sa(agent, obs_space, action_space, state_shape, config)
    if model_path != "":
        lagent.load_model(model_path)

    # Test and log, plot results
    test_episodes = eps

    test = create_tester_sa(agent)
    rewards = test(env, lagent, test_episodes)
    env.close()

    avg = np.sum(rewards) / test_episodes
    print("Avg. Test Reward " + lagent.__repr__(), avg)

    if not viz:
        plot_reward_histograms_sa(rewards, lagent.__repr__())


# ------------------------- PettingZoo Environments -------------------------


def create_agent_ma(
    agent, obs_space, action_space, config: Dict, dummy: bool, parent_agent=None
):
    if agent == "DQN":
        if not dummy:
            return DQNAgentMA(obs_space, action_space, config["hidden_layer_size"])
        else:
            return DQNAgentMA_Dummy(
                obs_space, action_space, config["hidden_layer_size"], parent_agent
            )

    elif agent == "PGN":
        if not dummy:
            return PGNAgentMA(
                obs_space, action_space, config["hidden_layer_size"], dummy
            )
        else:
            return PGNAgentMA_Dummy(
                obs_space, action_space, config["hidden_layer_size"], parent_agent
            )

    elif agent == "Random":
        return RandomAgentMA(id="gym")
    else:
        raise Exception("Invalid agent name")


def create_trainer_ma(agent):
    trainer = train_on_policy_ma
    if agent == "PGN":
        trainer = train_pgm_ma
    return trainer


def create_tester_ma(agent):
    tester = test_ma
    if agent == "PGN":
        tester = test_pgm_ma
    return tester


class PettingZooEnvs(str, Enum):
    t3 = "t3"
    texas_holem = "texas-holdem"


def make_petting_zoo_env(env_name, viz):
    from pettingzoo.classic import texas_holdem_v4, tictactoe_v3

    # env = texas_holdem_v4.env(render_mode="human")
    if viz:
        render_mode = "human"
    else:
        render_mode = None
    if env_name == "texas-holdem":
        env = texas_holdem_v4.env(render_mode=render_mode)
        obs_space = 72
        action_space = 4
    elif env_name == "t3":
        env = tictactoe_v3.env(render_mode=render_mode)
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
    retrain: bool = False,
    model: str = "",
    agent: str = "DQN",
    hidden_size: int = 64,
    num_players: int = 2,
):
    model_path = model
    env, obs_space, action_space = make_petting_zoo_env(env_name, False)

    config = {
        "num_players": num_players,
        "hidden_layer_size": hidden_size,
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

    lagent = create_agent_ma(agent, obs_space, action_space, config, False)
    ragents = [create_agent_ma(agent, obs_space, action_space, config, True, lagent)]
    # ragents = [RandomAgentMA("r" + str(_)) for _ in range(env.max_num_agents - 1)]
    agents = [lagent, *ragents]

    if retrain:
        lagent.load_model(model_path)

    # callback function for saving the model
    def save_model(episode):
        if episode % 200_000 == 0:
            lagent.save_model(f"../models/{task_name}-{episode}")

    train = create_trainer_ma(agent)
    train(env, agents, config["episodes"], on_episode_complete=save_model)

    # Test and log, plot results
    test_episodes = 1000
    test = create_tester_ma(agent)
    rewards = test(env, agents, test_episodes)
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
    model: str = "",
    viz: bool = False,
    hidden_size: int = 64,
    num_players: int = 2,
):
    model_path = model
    env, obs_space, action_space = make_petting_zoo_env(env_name, viz)

    config = {
        "num_players": num_players,
        "hidden_layer_size": hidden_size,
        "episodes": eps,
    }

    lagent = create_agent_ma(agent, obs_space, action_space, config, False)
    ragents = [RandomAgentMA(id="r" + str(_)) for _ in range(env.max_num_agents - 1)]
    agents = [lagent, *ragents]

    if model_path != "":
        lagent.load_model(model_path)

    # Test and log, plot results
    test_episodes = eps
    test = create_tester_ma(agent)
    rewards = test(env, agents, test_episodes)

    env.close()

    for idx, agent in enumerate(agents):
        avg = np.sum(rewards[idx]) / test_episodes
        print("Avg. Test Reward " + agent.__repr__(), avg)

    if not viz:
        plot_reward_histograms_ma(rewards, [agent.__repr__() for agent in agents])


# -------------------------   Probe  Environments   -------------------------


@app.command("probe1_ma")
def probe1ma_command():
    lagent = DQNAgentMA(1, 1, 4)
    lagent.update_batch_size = 1
    p1 = ProbeEnv1()
    train_on_policy_ma(p1, [lagent], 2000)

    value = lagent._value([0], 0).item()
    ok = abs(value - 1) < 0.01

    print(f"Expected value = 1 Learned = {value}")
    print("OK" if ok else "Not OK")

    if not ok:
        exit(1)


if __name__ == "__main__":
    app()
