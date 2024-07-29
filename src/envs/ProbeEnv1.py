import gymnasium

from .Env import Env, AgentID, ObsType, ActionType


class ProbeEnv1(Env):
    """
    Single Player, One state transition, Reward = 1 always
    """

    def __init__(self):
        self.name = "probe1"
        self.possible_agents = [0]
        self.render_mode = None
        self._state = -1

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(1)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(1)

    def reset(self):
        self._state = 0
        self.agent_selection = 0
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent: AgentID) -> ObsType:
        return {"obs": [self._state], "action_mask": [1]}

    def step(self, action: ActionType):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        assert action == 0
        self._state = 1
        self.rewards[0] = 1
        self.terminations[0] = True
        self._cumulative_rewards[0] = 1

    def __repr__(self):
        out = f"Environment: {self.name}\n" if self.name else "Environment: Unknown\n"
        return out
