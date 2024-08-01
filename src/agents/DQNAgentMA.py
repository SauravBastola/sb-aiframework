import os
from typing import Any
import torch
import torch.nn as nn
import numpy as np

from envs import ObsType, ActionType
from utils import ensure_path_parents

from .Agent import Agent
from .RandomAgentMA import random_choice


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __repr__(self) -> str:
        return f"{self.fc1.in_features} -> {self.fc2.in_features} -> {self.fc3.in_features} -> {self.fc3.out_features})"


class DQNAgentMA(Agent):
    TRAIN: bool = True

    def __init__(self, state_input_size, action_input_size, hidden_size, eps=0.1):
        self.action_input_size = action_input_size
        self.q_net = Network(state_input_size + action_input_size, hidden_size, 1)
        self.optim = torch.optim.Adam(self.q_net.parameters())
        self.prev_sa = None
        self.gamma = 1
        self.loss = torch.tensor([0], dtype=torch.float32)
        self.updates = 0
        self.update_batch_size = 32

        self.eps = eps

    def _encode_sa(self, s: list[int], a):
        action = torch.zeros(self.action_input_size, dtype=torch.float32)
        action[a] = 1
        return torch.concat([torch.tensor(s, dtype=torch.float32), action])

    def _value(self, s, a):
        return self.q_net.forward(self._encode_sa(s, a))

    def get_action(self, obs: ObsType):
        s = np.array(obs["observation"]).flatten()
        action_mask = obs["action_mask"]
        valid_actions = valid_actions_from_action_mask(action_mask)

        if self.TRAIN and np.random.random() < self.eps:
            return np.random.choice(valid_actions)

        best_value = self._value(s, valid_actions[0])
        best_action = valid_actions[0]

        for action in valid_actions[1:]:
            value = self._value(s, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def update(
        self,
        obs: ObsType,
        r: float,
        a: ActionType,
        terminate: bool,
        truncate: bool,
        info: Any,
    ):
        done = terminate or truncate
        s = np.array(obs["observation"]).flatten()

        if self.prev_sa:
            s_prev, a_prev = self.prev_sa
            target = r if done else (r + self.gamma * self._value(s, a)).item()
            pred = self._value(s_prev, a_prev)
            diff = pred - target

            self.loss += diff**2
            self.updates += 1

            if self.updates % self.update_batch_size == 0:
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()
                self.loss = torch.tensor([0], dtype=torch.float32)

        if done:
            self.prev_sa = None
        else:
            self.prev_sa = (s, a)

    def save_model(self, path):
        ensure_path_parents(path)
        torch.save(self.q_net.state_dict(), path + ".pth")
        print("Saved model", path)

    def load_model(self, path):
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path))
            print("Loaded model", path)
        else:
            print("Couldn't find model!")

    def __repr__(self):
        return f"DQNAgent({self.q_net.__repr__()})"


def valid_actions_from_action_mask(action_mask: list[int]):
    valid_actions = []
    for action, valid in enumerate(action_mask):
        if valid == 1:
            valid_actions.append(action)

    return valid_actions
