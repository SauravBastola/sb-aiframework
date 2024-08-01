import os
from typing import Any
import torch
import torch.nn as nn
import numpy as np

from utils import ensure_path_parents

from .Agent import Agent


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


class DQNAgentSA2(Agent):
    TRAIN: bool = True

    def __init__(
        self,
        state_input_size,
        action_input_size,
        hidden_size,
        state_shape=None,
        eps=0.1,
    ):
        self.action_input_size = action_input_size
        self.q_net = Network(state_input_size, hidden_size, action_input_size)
        self.optim = torch.optim.Adam(self.q_net.parameters())
        self.prev_sa = None
        self.gamma = 1
        self.loss = torch.tensor([0], dtype=torch.float32)
        self.updates = 0
        self.update_batch_size = 32
        self.state_shape = state_shape

        self.eps = eps

    def _value(self, s):
        return self.q_net.forward(torch.tensor(s, dtype=torch.float32))

    def get_action(self, obs, actions):
        if self.state_shape:
            li = []
            for idx, val in enumerate(self.state_shape):
                arr = np.zeros(val, dtype=np.float32)
                if isinstance(obs, int):
                    arr[obs] = 1.0
                else:
                    arr[obs[idx]] = 1.0
                li.append(arr)

            obs = li
        else:
            obs = [np.array(obs)]

        s = np.concatenate(obs)

        if self.TRAIN and np.random.random() < self.eps:
            return np.random.choice(actions)

        best_action = torch.argmax(self._value(s)).item()
        return best_action

    def update(self, obs, action, r, obs_next, action_next, done):
        if self.state_shape:
            li = []
            for idx, val in enumerate(self.state_shape):
                arr = np.zeros(val, dtype=np.float32)
                if isinstance(obs, int):
                    arr[obs] = 1.0
                else:
                    arr[obs[idx]] = 1.0
                li.append(arr)

            obs = li

            li_next = []
            for idx, val in enumerate(self.state_shape):
                arr = np.zeros(val, dtype=np.float32)
                if isinstance(obs_next, int):
                    arr[obs_next] = 1.0
                else:
                    arr[obs_next[idx]] = 1.0
                li_next.append(arr)
            obs_next = li_next
        else:
            obs = [np.array(obs)]
            obs_next = [np.array(obs_next)]

        s = np.concatenate(obs)
        s_next = np.concatenate(obs_next)
        target = (
            # r if done else (r + self.gamma * self._value(s_next, action_next)).item()
            r
            if done
            else r + self.gamma * self._value(s_next)[action_next].item()
        )
        q_value = self._value(s)
        q_value[action] = target

        self.loss += nn.MSELoss()(
            torch.tensor(q_value.detach().numpy(), dtype=torch.float32), self._value(s)
        ).mean()
        self.updates += 1

        if self.updates % self.update_batch_size == 0:
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()
            self.loss = torch.tensor([0], dtype=torch.float32)

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
