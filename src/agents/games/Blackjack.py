import numpy as np

from envs import ObsType, ActionType

from agents.Agent import Agent

card_val = [11, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2]


class BlackjackBaseAgent(Agent):
    def __init__(self):
        self.eps = None
        pass

    def get_action(self, obs: ObsType) -> ActionType:
        action_mask = obs["action_mask"]
        state = obs["obs"]

        actions = []
        for action, mask in enumerate(action_mask):
            if mask == 1:
                actions.append(action)

        if 1 in actions:
            return 1

        player_sum = 0

        for i, v in enumerate(state[: 52 * 2]):
            if v == 1:
                player_sum += card_val[i % 13]

        if 4 in actions and player_sum == 22:
            return 4

        if player_sum >= 17:
            assert 3 in actions
            return 3
        else:
            assert 2 in actions
            return 2

    def update(self, *args):
        pass

    def __repr__(self):
        return "BlackJack base agent"
