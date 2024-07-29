import pettingzoo as zoo
import gymnasium
from typing import Dict, Optional, Tuple, TypedDict

AgentID = int
ObsType = TypedDict(
    "ObsType",
    {"obs": list[int], "action_mask": list[int], "full_state": Optional[list[int]]},
)
ActionType = int | None  # Action is none for dead agents

Env = zoo.AECEnv[AgentID, ObsType, ActionType]
