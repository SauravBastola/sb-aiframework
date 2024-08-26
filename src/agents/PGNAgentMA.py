import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

from .DQNAgentMA import valid_actions_from_action_mask
from envs import ObsType

from .Agent import Agent

from collections import deque

from utils import ensure_path_parents

import copy
from .RandomAgentMA import random_choice


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


class PGNAgentMA(Agent):
    TRAIN: bool = True

    def __init__(
        self,
        state_input_size,
        action_input_size,
        hidden_size,
        state_shape=None,
        eps=0.1,
        dummy=False,
    ):
        self.action_input_size = action_input_size
        self.policy_net = PolicyNet(state_input_size, hidden_size, action_input_size)
        self.optim = torch.optim.Adam(self.policy_net.parameters())
        self.gamma = 1
        self.scores_deque = deque(maxlen=100)
        self.scores = []

    def get_action(self, obs: ObsType):
        s = np.array(obs["observation"])
        action_mask = obs["action_mask"]
        valid_actions = valid_actions_from_action_mask(action_mask)
        state = torch.tensor(s).flatten().float().unsqueeze(0)
        probs = self.policy_net.forward(state)
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)

    def update(self, log_probs, rewards):
        discounts = [self.gamma**i for i in range(len(rewards) + 1)]
        G_tmp = [a * b for a, b in zip(discounts, rewards)]
        G = []
        for i in range(len(G_tmp)):
            G.append(sum(G_tmp[i:]))

        policy_loss = torch.tensor([0.0])
        for g, log_prob in zip(G, log_probs):
            policy_loss += -log_prob * g

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
        return f"PGNAgentMA({self.policy_net.__repr__()})"


class PGNAgentMA_Dummy(PGNAgentMA):
    TRAIN: bool = False

    def __init__(
        self,
        state_input_size,
        action_input_size,
        hidden_size,
        parent_agent,
        follow_rate=1000,
    ):
        self.action_input_size = action_input_size
        self.policy_net = None
        self.parent_agent = parent_agent
        self.num_episodes = 0
        self.follow_rate = follow_rate

    def update(self, log_probs, rewards):
        self.num_episodes += 1

        if self.num_episodes >= self.follow_rate:
            self.num_episodes = 0
            self.copy_model()

    def get_action(self, obs: ObsType):
        if not self.policy_net:
            action_mask = obs["action_mask"]
            return random_choice(action_mask), 1
        else:
            return super().get_action(obs)

    def copy_model(self):
        self.policy_net = copy.deepcopy(self.parent_agent.policy_net)

    def save_model(self):
        pass

    def __repr__(self):
        return f"PGNAgentMA_Dummy :)"
