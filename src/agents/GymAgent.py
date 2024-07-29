from __future__ import annotations
import gymnasium
from typing import Any
from agents import Agent
import queue
import threading

from envs.Env import ObsType, ActionType


class SafeInterruptError(Exception):
    pass


def wait_for(queue: queue.Queue, event: str) -> Any:
    e = queue.get()
    if e[0] == "interrupt":
        raise SafeInterruptError()

    if e[0] != event:
        raise Exception("Unexpected event", e, "Expected", event)

    return e[1]


def interrupt_gym_agent(agent: GymAgent):
    agent.env_msg_queue.put_nowait(("interrupt",))


class GymAgent(gymnasium.Env[ObsType, ActionType], Agent):
    def __init__(
        self, observation_space: gymnasium.Space, action_space: gymnasium.Space
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_msg_queue = queue.Queue()
        self.env_msg_queue = queue.Queue()

    def start_train_loop(self):
        raise Exception(
            "Subclass of GymAgent must implement this method. This method should start the training loop"
        )

    def stop(self):
        interrupt_gym_agent(self)

    def _start(self):
        try:
            self.start_train_loop()
        except SafeInterruptError:
            pass

    def start(self):
        thread = threading.Thread(target=lambda: self._start())
        thread.start()

    ## gymnasium.Env Methods
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        while True:
            observation, reward, terminated, truncated, info = wait_for(
                self.env_msg_queue, "observation"
            )
            if terminated or truncated:
                pass  # We wait for another episode
            else:
                assert (
                    reward == 0
                ), "Can't handle non-zero reward at beginning of agent cycle"
                return observation, info

    def step(self, action: ActionType):
        if action != None:
            self.agent_msg_queue.put_nowait(("action", action))

        observation, reward, terminated, truncated, info = wait_for(
            self.env_msg_queue, "observation"
        )
        return observation, reward, terminated, truncated, info

    ## AGENT Methods

    def get_action(self, obs: ObsType) -> int:
        if self.TRAIN == False:
            raise Exception(
                "Subclass of GymAgent must implement get_action for self.TRAIN = False"
            )

        act = wait_for(self.agent_msg_queue, "action")
        return act

    def observe(
        self, obs: ObsType, reward: float, terminate: bool, truncate: bool, info: Any
    ) -> None:
        self.env_msg_queue.put_nowait(
            ("observation", (obs, reward, terminate, truncate, info))
        )
