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
			'max_travel_distance': 150,
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
	'measurement_distance': 2,
	'number_of_actions': 8,
	'kernel_length_scale': 2,
	'random_benchmark': True,
	'observation_type': 'visual',
	'max_collisions': 10,
	'eval_mode': True,
	'seed': 10,
	'reward_type': 'improvement',
}

N_EVAL = 1
NUM_OF_INDIVIDUALS = 20
NUM_OF_GENERATIONS = 10

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

	env.reset()
	done_flag = False
	individual = []

	while not done_flag:
		actions = []

		for veh_id in range(N):
			mask = env.get_action_mask(ind=veh_id).astype(int)
			action = np.random.choice(np.arange(0, 8), p=mask / np.sum(mask))
			actions.append(action)

		actions_dict = {i: a for i, a in enumerate(actions)}
		individual.extend(actions)
		_, _, done, _ = env.step(actions_dict)

		done_flag = np.all(list(done.values()))

	return creator(individual)


# --- Create the Genetic Algorihtm Strategies --- #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # This means we want to maximize
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)  # Individual creator
# Create a toolbox for genetic operations #
toolbox = base.Toolbox()
# Each individual is a set of n_agents x 101 steps (this will depend on the number of possible actions for agent)  #
toolbox.register("individual", createValidIndividual, creator.Individual)
# Create the population creator #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalEnv(ind, local_env):
	""" Evaluate the environment N_EVAL times. The individual is split into N agents actions
    and each agent is evaluated on the environment. """

	individual = np.array(ind)
	individual = individual.reshape(-1, N)  # Reshape the individual to be N agents x 101 steps

	for _ in range(N_EVAL):
		# Evaluate the individual N_EVAL times #
		local_env.reset()
		R = []

		for t in range(len(individual)):
			# Take the actions at instant t and create a dictionary of actions #
			actions = {i: a for i, a in enumerate(individual[t, :])}
			_, rewards, dones, _ = local_env.step(actions)

			R.append(np.sum(list(rewards.values())))

	return np.mean(R),


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)


# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)


def optimize(local_env, save=False):
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
	eaSimpleWithReevaluation(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=NUM_OF_GENERATIONS, stats=stats, halloffame=hof)

	if save:

		with open(f"ga_simple_optimization_result_{datetime.datetime.now().strftime('%Y_%m_%d-%H:%M_%S')}.txt", "w") as solution_file:

			solution_file.write("Optimization result for the GA\n")
			solution_file.write("---------------------------------\n")
			solution_file.write("---------------------------------\n")
			solution_file.write("--------- Best Individuals -----------\n")

			for idx, individual in enumerate(hof):
				str_data = ','.join(str(i) for i in individual)
				solution_file.write(f"Individual {idx}: {str_data}\n")
				solution_file.write(f"Fitness: {individual.fitness.values}\n")

			solution_file.close()

	return hof[0]


if __name__ == "__main__":

	np.random.seed(0)

	# Initial reset #

	env.reset()

	""" OPTIMIZE THE SCENARIO """
	best = np.asarray(optimize(env, save=True))
	individual = np.asarray(best).reshape(-1, N)

	env.eval()

	for run in range(N_EVAL):
		# Evaluate the individual N_EVAL times #
		env.reset()
		R = 0

		for t in range(len(individual)):
			# Take the actions at instant t and create a dictionary of actions #
			actions = {i: a for i, a in enumerate(individual[t, :])}
			_, rewards, dones, _ = env.step(actions)

			R += np.sum(list(rewards.values()))

			# Register positions and metrics #
			metrics.register_step(run_num=run, step=t, metrics=[R, env.mse])
			for veh_id, veh in enumerate(env.fleet.vehicles):
				paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])

metrics.register_experiment()
paths.register_experiment()

# pool.close()
