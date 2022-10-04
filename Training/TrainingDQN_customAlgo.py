from Environment.IGEnvironments import InformationGatheringEnv
import numpy as np
from CustomDQNImplementation.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent


""" Environment configuration"""
navigation_map = np.genfromtxt('../Environment/wesslinger_map.txt')

N = 4

from ShekelGroundTruth import Shekel

gt = Shekel
gt_config_file = gt.sim_config_template
gt_config_file['navigation_map'] = navigation_map

env_config = {'fleet_configuration': {
	'vehicle_configuration': {
		'dt': 0.1,
		'navigation_map': navigation_map,
		'target_threshold': 0.5,
		'ground_truth': np.random.rand(50, 50),
		'measurement_size': np.array([0, 0]),
		'max_travel_distance': 50,
	},
	'number_of_vehicles': N,
	'initial_positions': np.array([[15, 19],
	                               [13, 19],
	                               [18, 19],
	                               [15, 22]])
}, 'movement_type': 'DIRECTIONAL',
	'navigation_map': navigation_map,
	'min_measurement_distance': 3,
	'max_measurement_distance': 6,
	'measurement_distance': 3,
	'number_of_actions': 8,
	'kernel_length_scale': (12.27, 12.27, 50),
	'kernel_length_scale_bounds': ((0.1, 30), (0.1, 30), (0.001, 100)),
	'random_benchmark': True,
	'observation_type': 'visual',
	'max_collisions': 5,
	'eval_mode': False,
	'seed': 23,
	'reward_type': 'improvement',
	'dynamic': False,
	'ground_truth': gt,
	'ground_truth_config': gt_config_file,
	'temporal': False,
	'full_observable': False}

env = InformationGatheringEnv(env_config)

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
	logdir='./custom_runs',
	log_name="Uncertainty_experiment",
	save_every=1000,
	train_every=1,
)

agent.train(50_000)