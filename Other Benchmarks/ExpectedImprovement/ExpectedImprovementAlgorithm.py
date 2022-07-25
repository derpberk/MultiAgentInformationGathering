""" Algorithm that chooses next acquisition points
using the expected improvement reward in a BO style. """

import numpy as np
from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from skopt.acquisition import gaussian_ei
import matplotlib.pyplot as plt

navigation_map = np.genfromtxt('../../Environment/wesslinger_map.txt')

N = 4

env_config = {
	'fleet_configuration': {
		'vehicle_configuration': {
			'dt': 0.5,
			'navigation_map': navigation_map,
			'target_threshold': 0.5,
			'ground_truth': np.random.rand(50,50),
			'measurement_size': np.array([2, 2])
		},
		'number_of_vehicles': N,
		'max_distance': 100,
		'initial_positions': np.array([[15, 19],
		                               [13, 19],
		                               [18, 19],
		                               [15, 22]])
	},
	'movement_type': 'DIRECTIONAL',
	'navigation_map': navigation_map,
	'dynamic': 'Shekel',
	'min_measurement_distance': 5,
	'max_measurement_distance': 10,
	'measurement_distance': 3,
	'number_of_actions': 8,
	'kernel_length_scale': 2
}

# Create the environment #
env = InformationGatheringEnv(env_config=env_config)

np.random.seed(0)

# Create the evaluator and pass the metrics #
evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'MSE'], algorithm_name='Greedy ExpImp', experiment_name='ExpectedImprovementResults')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Greedy ExpImp',
                           experiment_name='ExpectedImprovementResults_paths',
                           directory='./')

def predict_best_action(ei_map):
	""" Compute the actions to move towards the maximum scaling ei_map position """

	actions = []

	for idx, vehicle in enumerate(env.fleet.vehicles):
		# Compute the position of the highest value of the ei_map that is at a distance env_config['measurement_distance'] form the vehicle current
		# position #



		look_ahead_distance = env_config['measurement_distance']*2
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

		plt.plot(ei_position[1], ei_position[0], 'ro')

		# Compute the angle to the highest value of the ei_map #
		angle = np.arctan2(ei_position[1]-vehicle.position[1], ei_position[0]-vehicle.position[0])

		angle = angle + 2 * np.pi if angle < 0 else angle

		# Compute the action to move towards the highest value of the ei_map #

		error = np.abs(angle - np.arange(0, env_config['number_of_actions'])/env_config['number_of_actions'] * 2*np.pi)
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

# Evaluate for 10 scenarios #
for run in range(30):

	# Reset flags
	t, R, done_flag = 0, 0, False

	# Initial reset #
	env.reset()

	expected_improvement_map = np.copy(navigation_map)

	print("Run Nº ", run)

	while not done_flag:

		# ---- Optimization routine ---- #
		# 1. Compute the Expected Improvement map #
		expected_improvement_values = gaussian_ei(env.visitable_positions, y_opt=0, model=env.GaussianProcess, xi=1)
		expected_improvement_map[env.visitable_positions[:,0], env.visitable_positions[:,1]] = expected_improvement_values
		# 2. Compute the actions
		dict_actions = predict_best_action(expected_improvement_map)

		_, reward_dict, dones_dict, info = env.step(dict_actions)

		# Accumulate the reward into the fitness #
		R += np.sum(list(reward_dict.values()))
		# Check if done #
		done_flag = any(list(dones_dict.values()))

		metrics = [R, env.mse]

		evaluator.register_step(run_num=run, step=t, metrics=[*metrics])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])

		t += 1

		# env.render()



evaluator.register_experiment()
paths.register_experiment()