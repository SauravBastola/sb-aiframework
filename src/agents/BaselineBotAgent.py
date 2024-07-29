from envs import ActionType, ObsType, BhoosGameEnv
from .Agent import Agent

from utils import req


class BaselineBotAgent(Agent):
    def __init__(self, bhoos_env: BhoosGameEnv):
        super().__init__()
        self.env = bhoos_env

    def get_action(self, obs: ObsType) -> ActionType:
        bot_action = req(
            f"baseline-bot-action/{self.env.instance_id}/{self.env.agent_selection}"
        )
        return bot_action

    def update(self, *args):
        pass

    def __repr__(self):
        return "BaselineBotAgent"
