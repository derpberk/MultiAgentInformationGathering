from Environment.EnvironmentCreation import generate_FF_env
import numpy as np
from CustomDQNImplementation.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent

# Generate the conditions #
reward_type = "KL"

env = generate_FF_env(reward_type=reward_type)

""" Set the configuration for the training """
# Create our RLlib Trainer.
agent = MultiAgentDuelingDQNAgent(
	env=env,
	memory_size=500_000,
	batch_size=64,
	target_update=1000,
	epsilon_values=[1.0, 0.05],
	epsilon_interval=[0.0, 0.5],
	learning_starts=10,
	gamma=0.95,
	lr=1e-4,
	# PER parameters
	alpha=0.2,
	beta=0.6,
	prior_eps=1e-6,
	# NN parameters
	number_of_features=512,
	logdir='./runs',
	log_name=f"Firefront_DDQL_{reward_type}_experiment",
	save_every=5000,
	train_every=1,
)

agent.train(50_000)