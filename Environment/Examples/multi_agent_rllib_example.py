import gym
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2

class FlexAgentsMultiAgent(MultiAgentEnv):
	"""Env of independent agents, each of which exits after n steps."""

	def __init__(self):
		super().__init__()
		self.agents = {}
		self._agent_ids = set()
		self.agentID = 0
		self.dones = set()
		self.observation_space = gym.spaces.Discrete(2)
		self.action_space = gym.spaces.Discrete(2)
		self.resetted = False

	def spawn(self):
		# Spawn a new agent into the current episode.
		agentID = self.agentID
		self.agents[agentID] = MockEnv(25)
		self._agent_ids.add(agentID)
		self.agentID += 1
		return agentID

	def reset(self):
		self.agents = {}
		self._agent_ids = set()
		self.spawn()
		self.resetted = True
		self.dones = set()
		obs = {}
		for i, a in self.agents.items():
			obs[i] = a.reset()

		return obs

	def step(self, action_dict):
		obs, rew, done, info = {}, {}, {}, {}
		# Apply the actions.
		for i, action in action_dict.items():
			obs[i], rew[i], done[i], info[i] = self.agents[i].process_action(action)
			if done[i]:
				self.dones.add(i)

		# Sometimes, add a new agent to the episode.
		if random.random() > 0.75 and len(action_dict) > 0:
			i = self.spawn()
			obs[i], rew[i], done[i], info[i] = self.agents[i].process_action(action)
			if done[i]:
				self.dones.add(i)

		# Sometimes, kill an existing agent.
		if len(self.agents) > 1 and random.random() > 0.25:
			keys = list(self.agents.keys())
			key = random.choice(keys)
			done[key] = True
			del self.agents[key]

		done["__all__"] = len(self.dones) == len(self.agents)
		return obs, rew, done, info



if __name__ == '__main__':


	env = FlexAgentsMultiAgent()

