import threading, os
import torch, torch.nn as nn
import tianshou as ts
import gymnasium, gymnasium.spaces as spaces

from envs.Env import ObsType, ActionType

from .GymAgent import GymAgent, SafeInterruptError, interrupt_gym_agent
from utils import ensure_path_parents, index_of_max


class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_size),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class TSDQNAgent(GymAgent):
    def __init__(
        self, observation_space: spaces.Discrete, action_space: spaces.Discrete
    ) -> None:
        super().__init__(observation_space, action_space)

        # Net
        self.net = Net(observation_space.n, action_space.n)
        optim = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # Policy
        self.policy = ts.policy.DQNPolicy(
            model=self.net,
            optim=optim,
            action_space=action_space,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    def start_train_loop(self):
        # Collector
        train_collector = ts.data.Collector(
            self.policy,
            self,
            ts.data.VectorReplayBuffer(20000, 10),
            exploration_noise=True,
        )

        # Trainer
        self.trainer = ts.trainer.OffpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            max_epoch=10,
            step_per_epoch=10000,
            step_per_collect=10,
            update_per_step=0.1,
            episode_per_test=100,
            batch_size=64,
        ).run()

    def get_action(self, obs: ObsType) -> int:
        if self.TRAIN:
            return super().get_action(obs)
        else:
            logits, _ = self.net.forward([obs["obs"]])
            action_mask = obs["action_mask"]

            return index_of_max(logits[0], action_mask)

    ## Transform observations

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = super().reset(seed=seed)
        return transform_obs(obs), info

    def step(self, action: ActionType):
        obs, *rest = super().step(action)
        return transform_obs(obs), *rest

    ## Model saving
    def save_model(self, path):
        ensure_path_parents(path)
        torch.save(self.net.state_dict(), path + ".pth")
        print("Saved model", path)

    def load_model(self, path):
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path))
            print("Loaded model", path)
        else:
            print("Couldn't find model!")

    def __repr__(self):
        return f"TS_DQNAgent"


def transform_obs(obs: ObsType):
    return {"obs": obs["obs"], "mask": obs["action_mask"]}
