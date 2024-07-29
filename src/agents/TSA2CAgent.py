import os
import torch, torch.nn as nn
import tianshou as ts
import gymnasium.spaces as spaces
from torchrl import torchrl
from typing import Any, Optional, Tuple

from envs.Env import ObsType, Env
from envs.BhoosGameEnv import BhoosGameEnv

from .GymAgent import GymAgent
from utils import ensure_path_parents, index_of_max


class PolicyNet(nn.Module):
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

    def forward(self, observation, state=None, info={}):
        obs = observation["obs"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        mask = torch.tensor(observation["action_mask"], dtype=torch.int) == 1
        return (logits, mask), state


class CriticNet(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, observation, state=None, info={}):
        obs = (
            observation["full-state"]
            if "full-state" in observation
            else observation["obs"]
        )
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        value = self.model(obs.view(batch, -1))
        return value


class TSA2CAgent(GymAgent):
    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
        use_full_state_critic: Optional[Tuple[BhoosGameEnv, spaces.Discrete]] = None,
    ) -> None:
        super().__init__(observation_space, action_space)

        if use_full_state_critic:
            critic_observation_space = use_full_state_critic[1]
            self.critic_full_state_fn = lambda: use_full_state_critic[
                0
            ].observe_full_state()
        else:
            critic_observation_space = observation_space
            self.critic_full_state_fn = None

        # Net
        self.actor_net = PolicyNet(observation_space.n, action_space.n)
        self.critic_net = CriticNet(critic_observation_space.n)
        optim = torch.optim.Adam(
            [
                *self.actor_net.parameters(),
                *self.critic_net.parameters(),
            ],
            lr=1e-3,
        )

        # Policy
        self.policy = ts.policy.A2CPolicy(
            actor=self.actor_net,
            critic=self.critic_net,
            dist_fn=lambda logits, mask: torchrl.modules.MaskedCategorical(
                logits=logits, mask=mask
            ),
            optim=optim,
            action_space=action_space,
            action_scaling=False,
        )

    def start_train_loop(self):
        # Collector
        train_collector = ts.data.Collector(
            self.policy,
            self,
            ts.data.ReplayBuffer(20000),
            exploration_noise=True,
        )

        # Trainer
        self.trainer = ts.trainer.OnpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            max_epoch=10,
            repeat_per_collect=1,
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
            out, _ = self.actor_net.forward(
                {"obs": [obs["obs"]], "action_mask": [obs["action_mask"]]}
            )
            logits, action_mask = out

            return index_of_max(logits[0], action_mask[0])

    def observe(
        self, obs: ObsType, reward: float, terminate: bool, truncate: bool, info: Any
    ) -> None:
        if self.critic_full_state_fn:
            obs["full-state"] = self.critic_full_state_fn()

        return super().observe(obs, reward, terminate, truncate, info)

    ## Model saving
    def save_model(self, path):
        ensure_path_parents(path)
        torch.save(self.actor_net.state_dict(), path + ".pth")
        print("Saved model", path)

    def load_model(self, path):
        if os.path.exists(path):
            self.actor_net.load_state_dict(torch.load(path))
            print("Loaded model", path)
        else:
            print("Couldn't find model!")

    def __repr__(self):
        return f"TS_A2CAgent"
