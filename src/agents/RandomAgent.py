import numpy as np

from envs import ObsType, ActionType

from .Agent import Agent


class RandomAgent(Agent):
    def __init__(self):
        self.eps = None
        pass

    def get_action(self, obs: ObsType) -> ActionType:
        action_mask = obs["action_mask"]
        return random_choice(action_mask)

    def update(self, *args):
        pass

    def __repr__(self):
        return "RandomAgent"


def random_choice(action_mask: list[int]):
    actions_count = 0
    for mask in action_mask:
        if mask == 1:
            actions_count += 1

    choice = np.random.randint(actions_count)
    count = 0
    for action, mask in enumerate(action_mask):
        if mask == 1:
            if count == choice:
                return action
            count += 1

    raise Exception(
        f"Couldn't choose random action. action_mask={action_mask}, choice={choice}"
    )
