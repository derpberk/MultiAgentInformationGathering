""" Algorithm that chooses next acquisition points
using the expected improvement reward in a BO style. """

import numpy as np
from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from skopt.acquisition import gaussian_ei
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

navigation_map = np.genfromtxt('../../Environment/wesslinger_map.txt')

N = 4

from ShekelGroundTruth import Shekel
from OilSpillEnvironment import OilSpill
from FireFront import WildfireSimulator

gt = Shekel
gt_config_file = gt.sim_config_template
gt_config_file['navigation_map'] = navigation_map

env_config = {
	'fleet_configuration': {
		'vehicle_configuration': {
			'dt': 0.1,
			'navigation_map': navigation_map,
			'target_threshold': 0.5,
			'ground_truth': np.random.rand(50, 50),
			'measurement_size': np.array([0, 0]),
			'max_travel_distance': 100,
		},
		'number_of_vehicles': N,
		'initial_positions': np.array([[15, 19],
		                               [13, 19],
		                               [18, 19],
		                               [15, 22]])
	},
	'movement_type': 'DIRECTIONAL',
	'navigation_map': navigation_map,
	'min_measurement_distance': 3,
	'max_measurement_distance': 6,
	'measurement_distance': 3,
	'number_of_actions': 8,
	'kernel_length_scale': (3.5, 3.5, 50),
	'kernel_length_scale_bounds': ((0.1, 10), (0.1, 10), (0.001, 100)),
	'random_benchmark': True,
	'observation_type': 'visual',
	'max_collisions': 10,
	'eval_mode': True,
	'seed': 23,
	'reward_type': 'improvement',
	'dynamic': False,
	'ground_truth': gt,
	'ground_truth_config': gt_config_file,
	'temporal': False,
}

# Create the environment #
env = InformationGatheringEnv(env_config=env_config)

np.random.seed(0)

# Create the evaluator and pass the metrics #
evaluator = MetricsDataCreator(metrics_names=['Mean Reward',
                                              'Average Uncertainty',
                                              'Mean regret',
                                              'Model Error'],
                               algorithm_name='Greedy ExpImp',
                               experiment_name='ExpectedImprovementResults')

paths = MetricsDataCreator(
							metrics_names=['vehicle', 'x', 'y'],
							algorithm_name='Greedy ExpImp',
							experiment_name='ExpectedImprovementResults_paths',
							directory='./')


def predict_best_action(ei_map):
	""" Compute the actions to move towards the maximum scaling ei_map position """

	actions = []

	for idx, vehicle in enumerate(env.fleet.vehicles):
		# Compute the position of the highest value of the ei_map that is at a distance env_config['measurement_distance'] form the vehicle current
		# position #

		look_ahead_distance = env_config['measurement_distance'] * 3
		look_ahead_mask = np.zeros_like(ei_map)

		px, py = vehicle.position.astype(int)

		# State - coverage area #
		x = np.arange(0, navigation_map.shape[0])
		y = np.arange(0, navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= look_ahead_distance ** 2
		look_ahead_mask[mask.T] = 1.0

		ei_index = np.argmax(ei_map * look_ahead_mask)

		ei_position = np.unravel_index(ei_index, ei_map.shape)

		# Compute the angle to the highest value of the ei_map #
		angle = np.arctan2(ei_position[1] - vehicle.position[1], ei_position[0] - vehicle.position[0])

		angle = angle + 2 * np.pi if angle < 0 else angle

		# Compute the action to move towards the highest value of the ei_map #

		error = np.abs(angle - 2 * np.pi * np.arange(0, env_config['number_of_actions']) / env_config['number_of_actions'])
		error[np.logical_not(env.get_action_mask(idx))] = np.inf


		action = np.argmin(error)
		# Set the action #
		actions.append(action)

	# Create a dictionary with every action (value) for every agent (key)#
	actions_dict = {idx: action for idx, action in enumerate(actions)}
	# Return the actions dictionary #
	return actions_dict

d = None

env.eval()

# Complimentary surrogate model

gp = GaussianProcessRegressor(kernel=RBF(length_scale=3.5, length_scale_bounds=(0.1, 100.0)), alpha=0.001)

# Evaluate for 10 scenarios #
for run in range(20):

	# Reset flags
	t, R, done_flag = 0, 0, False

	# Initial reset #
	env.reset()

	expected_improvement_map = np.copy(navigation_map)

	print("Run Nº ", run)


	while not done_flag:

		# ---- Optimization routine ---- #
		# 1. Compute the Expected Improvement map #
		expected_improvement_values = gaussian_ei(env.visitable_positions, model=env.GaussianProcess, xi=0.1)
		expected_improvement_map[env.visitable_positions[:, 0], env.visitable_positions[:, 1]] = expected_improvement_values
		# 2. Compute the actions
		dict_actions = predict_best_action(expected_improvement_map)

		_, reward_dict, dones_dict, info = env.step(dict_actions)

		# Accumulate the reward into the fitness #
		R += np.sum(list(reward_dict.values()))
		# Check if done #
		done_flag = any(list(dones_dict.values()))

		# Lets test if there is much more improvement with a complimentary surrogate model
		gp.fit(env.measured_locations, env.measured_values)

		surr_mu, surr_unc = gp.predict(env.visitable_positions, return_std=True)
		real_mu = env.ground_truth.ground_truth_field[env.visitable_positions[:,0], env.visitable_positions[:,1]]

		mse = mean_squared_error(y_true=real_mu, y_pred=surr_mu, squared=False)

		metrics = [info['metrics']['accumulated_reward'],
		           info['metrics']['uncertainty'],
		           info['metrics']['instant_regret'],
		           mse,
		           ]

		evaluator.register_step(run_num=run, step=t, metrics=[*metrics])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])

		t += 1

		# env.render()

evaluator.register_experiment()
paths.register_experiment()
