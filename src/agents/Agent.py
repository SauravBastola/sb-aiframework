from typing import Any
from envs import ObsType


# TODO: Define this as an abstract agent
class Agent:
    TRAIN: bool = True

    def observe(
        self, obs: ObsType, reward: float, terminate: bool, truncate: bool, info: Any
    ) -> None:
        pass

    def get_action(self, obs: ObsType) -> int:
        raise Exception("Agent get_action is not implemented")

    def update(
        self,
        obs: ObsType,
        reward: float,
        action: int | None,
        terminate: bool,
        truncate: bool,
        info: Any,
    ) -> None:
        pass
