import numpy as np
import matplotlib.pyplot as plt
import pyswarms.backend as P
from pyswarms.backend.topology import Star, Ring
import pyswarms as ps
from pyswarms.utils.plotters.formatters import Mesher

""" --- Environment --- """
from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

""" Compute Ground Truth """
navigation_map = np.genfromtxt('/Users/samuel/MultiAgentInformationGathering/Environment/wesslinger_map.txt')
N = 4
same_evaluation_scenario = True
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
		                               [23, 19],
		                               [18, 19],
		                               [15, 22]])
	},
	'movement_type': 'DIRECTIONAL',
	'navigation_map': navigation_map,
	'dynamic': 'Shekel',
	'min_measurement_distance': 3,
	'max_measurement_distance': 6,
	'measurement_distance': 3,
	'number_of_actions': 8,
	'kernel_length_scale': 2,
	'random_benchmark': True,
	'observation_type': 'visual',
	'max_collisions': 10,
	'eval_mode': True,
	'seed': 1000,
	'reward_type': 'improvement',
}

# Create the environment #
env = InformationGatheringEnv(env_config=env_config)

my_topology = Star() # The Topology Class
my_options = {'c1': 0, 'c2': 0.8, 'w': 0.4, 'c3': 0.4}  # arbitrarily set
my_swarm = P.create_swarm(n_particles=4, dimensions=2, options=my_options) # The Swarm Class

iterations = 100  # Set 100 iterations

def multi_evaluate_benchmark(swarm_positions):


	swarm_positions = np.clip(swarm_positions * navigation_map.shape, (0, 0), np.asarray(navigation_map.shape)-1).astype(int)
	values = [1 - env.ground_truth.read(position) for position in swarm_positions]

	return np.asarray(values)


position_history = []

my_swarm.position = env_config['fleet_configuration']['initial_positions']/navigation_map.shape

for i in range(iterations):


	# Part 1: Update personal best
	my_swarm.current_cost = multi_evaluate_benchmark(my_swarm.position)  # Compute current cost
	my_swarm.pbest_cost = multi_evaluate_benchmark(my_swarm.pbest_pos)  # Compute personal best pos
	my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm)  # Update and store
	position_history.append(my_swarm.position * navigation_map.shape)
	# Part 2: Update global best
	# Note that gbest computation is dependent on your topology
	if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
		my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

	# Let's print our output
	if i % 1 == 0:
		print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

	# Part 3: Update position and velocity matrices
	# Note that position and velocity updates are dependent on your topology
	my_swarm.velocity = my_topology.compute_velocity(my_swarm, clamp=(-0.1, 0.1))
	my_swarm.velocity = 0.05 * my_swarm.velocity/np.linalg.norm(my_swarm.velocity, axis=1).reshape(-1,1)
	# my_swarm.position = my_topology.compute_position(my_swarm)
	new_attempted_positions = my_topology.compute_position(my_swarm)

	valid = [bool(navigation_map[int(pos[0]*navigation_map.shape[0]), int(pos[1]*navigation_map.shape[1])]) for pos in new_attempted_positions]

	my_swarm.position[valid] = new_attempted_positions[valid]


plt.ion()
fig, ax = plt.subplots(1, 1)
plt.imshow(env.ground_truth.read())

d = ax.scatter(x=position_history[0][:,1], y=position_history[0][:,0], zorder=20, c='k')

for t in range(1, len(position_history)):

	position_history[t][:, [1, 0]] = position_history[t][:, [0, 1]]
	d.set_offsets((position_history[t]))

	fig.canvas.draw()

	plt.pause(0.1)
	plt.show()






