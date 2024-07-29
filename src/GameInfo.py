import gymnasium
from utils import req
from typing import Any, Optional

from SocketClient import SocketClient


def to_gym_space(space: Any):
    if isinstance(space, int):
        return gymnasium.spaces.Discrete(space)
    else:
        raise Exception("Unknown type of space", space)


class GameInfo:
    game_name: str
    name: str
    num_players: int
    observation_space: gymnasium.Space
    action_space: gymnasium.Space
    full_state_space: Optional[gymnasium.Space]

    def __init__(self, variant: Optional[str] = None):
        socket = SocketClient()
        info = socket.req("info")
        socket.close()

        variant = info["variants"][0] if variant == None else info["variants"][variant]
        self.game_name = info["name"]
        self.name = variant["name"]
        self.num_players = variant["players"]
        self.observation_space = to_gym_space(variant["observation_space"])
        self.action_space = to_gym_space(variant["action_space"])
        self.full_state_space = (
            to_gym_space(variant["full_state_space"])
            if "full_state_space" in variant
            else None
        )
