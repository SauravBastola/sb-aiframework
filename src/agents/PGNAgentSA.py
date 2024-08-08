import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

from .Agent import Agent

from collections import deque

from utils import ensure_path_parents


class PolicyNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def __repr__(self) -> str:
        return f"{self.fc1.in_features} -> {self.fc2.in_features} -> {self.fc3.in_features} -> {self.fc3.out_features})"


class PGNAgentSA(Agent):
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
        self.policy_net = PolicyNet(state_input_size, hidden_size, action_input_size)
        self.optim = torch.optim.Adam(self.policy_net.parameters())
        self.gamma = 1
        self.state_shape = state_shape
        self.scores_deque = deque(maxlen=100)
        self.scores = []

    def get_action(self, obs, actions=None):
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
        state = torch.tensor(s).float().unsqueeze(0)
        probs = self.policy_net.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)

    def update(self, log_probs, rewards):
        self.scores_deque.append(sum(rewards))
        self.scores.append(sum(rewards))

        discounts = [self.gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        # Backprogagation
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()

    def save_model(self, path):
        ensure_path_parents(path)
        torch.save(self.policy_net.state_dict(), path + ".pth")
        print("Saved model", path)

    def load_model(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            print("Loaded model", path)
        else:
            print("Couldn't find model!")

    def __repr__(self):
        return f"DQNAgent({self.policy_net.__repr__()})"
