import sys
sys.path.append('.')

from Environment.EnvironmentCreation import generate_WQ_env
import numpy as np
from CustomDQNImplementation.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent

# Generate the conditions #
reward_type = "KL"

env = generate_WQ_env(reward_type=reward_type)

""" Set the configuration for the training """

agent = MultiAgentDuelingDQNAgent(
	env=env,
	memory_size=500_000,
	batch_size=64,
	target_update=1000,
	epsilon_values=[1.0, 0.05],
	epsilon_interval=[0.0, 0.33],
	learning_starts=200,
	gamma=0.95,
	lr=1e-4,
	# PER parameters
	alpha=0.2,
	beta=0.6,
	prior_eps=1e-6,
	# NN parameters
	number_of_features=512,
	logdir='./runs',
	log_name=f"WQP_DDQL_{reward_type}_experiment",
	save_every=10000,
	train_every=5,
)

agent.train(100_000)