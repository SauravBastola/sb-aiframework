import numpy as np

from envs import ObsType, ActionType

from .Agent import Agent


class RandomAgentSA(Agent):
    def __init__(self, id):
        self.eps = None
        self.id = id
        pass

    def get_action(self, obs, actions) -> ActionType:
        return random_choice(actions)

    def update(self, *args):
        pass

    def __repr__(self):
        return "RandomAgent-" + self.id


def random_choice(actions):

    if len(actions) == 0:
        raise Exception(f"Actions is empty")

    choice = np.random.randint(len(actions))
    return actions[choice]
