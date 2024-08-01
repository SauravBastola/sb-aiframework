import pettingzoo as zoo
import gymnasium
from typing import Dict, Optional, Tuple, TypedDict

from utils import req

from .Env import Env, AgentID, ObsType, ActionType
from GameInfo import GameInfo

from SocketClient import SocketClient


class BhoosGameEnv(Env):
    """
    An environment that follows the interface of PettingZoo and interacts with bhoos games
    """

    game_variant: GameInfo
    instance_id: int = 0
    action_space_length: int

    def __init__(self, game_variant: GameInfo):
        """
        init method should define
        - possible_agents
        - render_mode
        """
        super().__init__()
        self.game_variant = game_variant

        action_space = self.game_variant.action_space
        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise Exception("ZooEnv only supports Discrete action space for now")

        self.action_space_length = int(action_space.n)
        # metadata holds environment constants
        # "render_modes" is from gymnasium
        self.metadata = {"render_modes": ["human"], "name": game_variant.name}

        self.possible_agents = [i for i in range(game_variant.num_players)]
        self.render_mode = "human"
        self.socket = SocketClient()

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.game_variant.observation_space

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.game_variant.action_space

    def render(self):
        print(f"Agents: {self.possible_agents}")

    def reset(self, seed=None):
        """
        Reset needs to initialize the following attributes
        - agent_selection
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        """
        if self.instance_id:
            self.socket.req(f"end-game/{self.instance_id}")
            self.instance_id = 0

        res = self.socket.req("start")
        self.instance_id = res["instanceId"]
        self.agent_selection = res["firstPlayer"]

        self.game_over = False

        self.agents = self.possible_agents.copy()
        assert (
            self.agent_selection in self.agents
        ), f"Agent Selection is {self.agent_selection}; Agents is {self.agents}"

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: AgentID) -> ObsType:
        print("I am called")
        res = self.socket.req(f"observe/{self.instance_id}/{agent}")
        state: list[int] = res["state"]
        valid_actions: list[int] = res["validActions"]
        action_mask = [0 for i in range(self.action_space_length)]
        for valid_action in valid_actions:
            action_mask[valid_action] = 1
        return {"observation": state, "action_mask": action_mask}

    def observe_full_state(self):
        full_state: list[int] = self.socket.req(
            f"observe-full-state/{self.instance_id}"
        )
        return full_state

    def step(self, action: ActionType) -> None:
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        # When the agent is terminated, its rewards, and state are observed once
        # and then the agent is step-ed over using action = None
        # we then cycle to next dead agent and so on.
        # _was_dead_step removes the dead agent from the agent list
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        self._cumulative_rewards[self.agent_selection] = 0.0

        res = self.socket.req(
            f"take-action/{self.instance_id}/{self.agent_selection}/{action}"
        )

        next_player: AgentID = res["next_player"]
        rewards: Optional[list[float]] = res["rewards"] if "rewards" in res else None
        game_over: bool = res["isTerminal"]

        if game_over:
            for agent in self.agents:
                self.terminations[agent] = True

        self.agent_selection = next_player if not game_over else self.agent_selection
        assert self.agent_selection != -1

        self._clear_rewards()
        if rewards:
            for agentIdx, reward in enumerate(rewards):
                self.rewards[agentIdx] = reward

        self._accumulate_rewards()

    def close(self):
        """
        Close rendering. Since we do no redering, no need to close
        """
        self.socket.close()
