import numpy as np
from deap import base
from deap import creator
from deap import tools
import random
import datetime
import multiprocessing
import matplotlib.pyplot as plt
from StochasticEASimple import eaSimpleWithReevaluation, cxTwoPointCopy
from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator


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
		                               [13, 19],
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
	'seed': 10,
	'reward_type': 'improvement',
}

N_EVAL = 10
NUM_OF_INDIVIDUALS = 20
NUM_OF_GENERATIONS = 10
INDIVIDUAL_MAX_SIZE = np.ceil(env_config['fleet_configuration']['vehicle_configuration']['max_travel_distance']/env_config['measurement_distance']) + 10

# Create the environment #
env = InformationGatheringEnv(env_config=env_config)

metrics = MetricsDataCreator(metrics_names=['Accumulated Reward', 'MSE'],
                             algorithm_name='Genetic Algorithm',
                             experiment_name='GeneticAlgorithm',
                             directory='./')

paths = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y'],
                           algorithm_name='Genetic Algorithm',
                           experiment_name='GeneticAlgorithm_paths',
                           directory='./')


def createValidIndividual(creator):
	""" Create a valid individual. """

	states = env.reset()
	done_flag = False
	done = {i:False for i in range(N)}
	done['__all__'] = False

	action_array = np.zeros((int(INDIVIDUAL_MAX_SIZE), N))
	t = 0

	while not done_flag:

		actions = []

		for veh_id in range(N):
			mask = env.get_action_mask(ind=veh_id).astype(int)
			action = np.random.choice(np.arange(0, 8), p=mask / np.sum(mask))
			actions.append(action)

		actions_dict = {i: a for i, a in enumerate(actions)}

		action_array[t, list(actions_dict.keys())] = np.asarray(list(actions_dict.values()))

		states, _, done, _ = env.step({i: actions_dict[i] for i in done.keys() if (not done[i]) and i != '__all__'})

		done_flag = np.all(list(done.values()))

		t+=1

	return creator(action_array.reshape(1, -1).astype(int)[0].tolist())

# --- Create the Genetic Algorihtm Strategies --- #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # This means we want to maximize
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)  # Individual creator
# Create a toolbox for genetic operations #
toolbox = base.Toolbox()
# Each individual is a set of n_agents x 101 steps (this will depend on the number of possible actions for agent)  #
toolbox.register("individual", createValidIndividual, creator.Individual)
# Create the population creator #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalEnv(list_ind, local_env):
	""" Evaluate the environment N_EVAL times. The individual is split into N agents actions
    and each agent is evaluated on the environment. """

	individual = np.array(list_ind)
	individual = individual.reshape(-1, N)  # Reshape the individual to be N agents x 101 steps

	for _ in range(N_EVAL):
		# Evaluate the individual N_EVAL times #
		local_env.reset()
		dones = {i: False for i in range(N)}
		dones['__all__'] = False
		R = []

		t = 0
		while not dones['__all__']:
			# Take the actions at instant t and create a dictionary of actions #
			actions_dict = {i: a for i, a in enumerate(individual[t, :])}
			states, rewards, dones, _ = local_env.step({i: actions_dict[i] for i in dones.keys() if (not dones[i]) and i != '__all__'})
			R.append(np.sum(list(rewards.values())))
			t+=1

			if any(local_env.number_of_collisions > 0.0):
				R = 0
				break

	return np.mean(R),


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)


# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)


def optimize(local_env):
	random.seed(64)

	pop = toolbox.population(n=NUM_OF_INDIVIDUALS)

	# Fix the evaluation function with the current environment
	toolbox.register("evaluate", evalEnv, local_env=local_env)

	# Hall of fame for the TOP 5 individuals #
	hof = tools.HallOfFame(5, similar=np.array_equal)

	# Statistics #
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	# Algorithm - Simple Evolutionary Algorithm #
	eaSimpleWithReevaluation(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=NUM_OF_GENERATIONS, stats=stats, halloffame=hof)

	return hof[0]


if __name__ == "__main__":

	np.random.seed(0)

	# Initial reset #

	env.reset()

	""" OPTIMIZE THE SCENARIO """
	best = np.asarray(optimize(env))
	individual = np.asarray(best).reshape(-1, N)

	env.eval()

	for run in range(N_EVAL):
		# Evaluate the individual N_EVAL times #
		env.reset()
		R = 0
		dones = {i: False for i in range(N)}
		dones['__all__'] = False
		t = 0

		while not dones['__all__']:
			# Take the actions at instant t and create a dictionary of actions #
			actions_dict = {i: a for i, a in enumerate(individual[t, :])}
			states, rewards, dones, _ = env.step({i: actions_dict[i] for i in dones.keys() if (not dones[i]) and i != '__all__'})
			R += np.sum(list(rewards.values()))
			t += 1

			# Register positions and metrics #
			metrics.register_step(run_num=run, step=t, metrics=[R, env.mse])
			for veh_id, veh in enumerate(env.fleet.vehicles):
				paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])


metrics.register_experiment()
paths.register_experiment()

# pool.close()
