import numpy as np
import matplotlib.pyplot as plt
import pyswarms.backend as P
from pyswarms.backend.topology import Star, Ring

""" --- Environment --- """
from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

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

"""
'initial_positions': np.array([[15, 19],
                               [13, 19],
                               [18, 19],
                               [15, 22]])
"""

env_config = {
	'fleet_configuration': {
		'vehicle_configuration': {
			'dt': 0.1,
			'navigation_map': navigation_map,
			'target_threshold': 0.5,
			'ground_truth': np.random.rand(50, 50),
			'measurement_size': np.array([0, 0]),
			'max_travel_distance': 50,
		},
		'number_of_vehicles': N,
		'initial_positions': np.array([[9, 17],
		                               [14, 40],
		                               [39, 41],
		                               [28, 7]]),
	},
	'movement_type': 'DIRECTIONAL',
	'navigation_map': navigation_map,
	'min_measurement_distance': 3,
	'max_measurement_distance': 6,
	'measurement_distance': 3,
	'number_of_actions': 8,
	'kernel_length_scale': (2.5, 2.5, 50),
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
                               algorithm_name='PSO',
                               experiment_name='PSOResults')

paths = MetricsDataCreator(
							metrics_names=['vehicle', 'x', 'y'],
							algorithm_name='PSO',
							experiment_name='PSOResults_paths',
							directory='./')

# Create the environment #
env = InformationGatheringEnv(env_config=env_config)

my_topology = Star() # The Topology Class
my_options = {'c1': 0.9, 'c2': 0.5, 'w': 0.1}  # arbitrarily set
my_swarm = P.create_swarm(n_particles=N, dimensions=2, options=my_options) # The Swarm Class
iterations = 100  # Set 100 iterations



env.eval()

# Complimentary surrogate model

gp = GaussianProcessRegressor(kernel=RBF(length_scale=15.0, length_scale_bounds=(2.0, 100.0)), alpha=0.001)


def velocity_to_action(vel_vec, idx):
	# Transform a speed vector into a directional vector #

	# Normalize the vector #
	angle = np.arctan2(vel_vec[1], vel_vec[0])
	angle = angle + 2 * np.pi if angle < 0 else angle

	error = np.abs(angle - 2 * np.pi * np.arange(0, env_config['number_of_actions']) / env_config['number_of_actions'])
	error[np.logical_not(env.get_action_mask(idx))] = np.inf

	return np.argmin(error)


for run in range(iterations):

	env.reset()
	my_swarm.position = env_config['fleet_configuration']['initial_positions'] / navigation_map.shape
	my_swarm = P.create_swarm(n_particles=N, dimensions=2, options=my_options)  # The Swarm Class

	done_flag = False

	# Reset flags
	t, R, done_flag = 0, 0, False

	while not done_flag:

		my_swarm.position = env.fleet.get_positions()

		# Part 1: Update personal best
		my_swarm.current_cost = 1.0 - np.asarray(list(map(env.ground_truth.read, my_swarm.position.astype(int))))  # Compute current cost
		my_swarm.pbest_cost = 1.0 - np.asarray(list(map(env.ground_truth.read, my_swarm.pbest_pos.astype(int))))  # Compute personal best pos
		my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)  # Update and store

		# Part 2: Update global best
		# Note that gbest computation is dependent on your topology
		if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
			my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

		# Part 3: Update position and velocity matrices
		# Note that position and velocity updates are dependent on your topology
		my_swarm.velocity = my_topology.compute_velocity(my_swarm)

		# Map the velocities into actions #
		actions = {i: velocity_to_action(my_swarm.velocity[i], i) for i in range(N)}

		_, reward_dict, dones_dict, info = env.step(actions)

		my_swarm.position = env.fleet.get_positions()

		# Accumulate the reward into the fitness #
		R += np.sum(list(reward_dict.values()))

		# Check if done #
		done_flag = any(list(dones_dict.values()))

		# Lets test if there is much more improvement with a complimentary surrogate model
		gp.fit(env.measured_locations, env.measured_values)

		surr_mu, surr_unc = gp.predict(env.visitable_positions, return_std=True)
		real_mu = env.ground_truth.ground_truth_field[env.visitable_positions[:, 0], env.visitable_positions[:, 1]]

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

		env.render()

	evaluator.register_experiment()
	paths.register_experiment()







