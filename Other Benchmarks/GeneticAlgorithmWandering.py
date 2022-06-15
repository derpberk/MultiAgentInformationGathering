from Environment.InformationGatheringEnvironments import SynchronousMultiAgentIGEnvironment
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import datetime
import multiprocessing
import matplotlib.pyplot as plt

plt.ion()

""" Compute Ground Truth """
navigation_map = np.genfromtxt('../Environment/wesslinger_map.txt')
n_agents = 3

agent_config = {'navigation_map': navigation_map,
                'mask_size': (1, 1),
                'initial_position': None,
                'speed': 2,
                'max_illegal_movements': 10,
                'max_distance': 200,
                'dt': 1}


my_env_config = {'number_of_agents':n_agents,
                 'number_of_actions': 8,
                 'kernel_length_scale': 2,
                 'agent_config': agent_config,
                 'navigation_map': navigation_map,
                 'meas_distance': 2,
                 'initial_positions': np.array([[21,14],[30,16],[36,41]]),
                 'max_meas_distance': 5,
                 'min_meas_distance': 1,
                 'dynamic': 'Shekel',
                 }

# Create the environment #
env = SynchronousMultiAgentIGEnvironment(env_config=my_env_config)


# Create the evaluator and pass the metrics #
evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Max regret', 'Collisions'], algorithm_name='GA', experiment_name='WanderingGAResults')

# --- Create the Genetic Algorihtm Strategies --- #
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # This means we want to maximize
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # Individual creator
# Create a toolbox for genetic operations #
toolbox = base.Toolbox()
# Cromosome is an action in [0,number_of_actions] #
toolbox.register("attr_bool", random.randint, 0, my_env_config['number_of_actions'])
# Each individual is a set of n_agents x 101 steps (this will depend on the number of possible actions for agent)  #
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=my_env_config['number_of_agents']*101)
# Create the population creator #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def evalEnv(ind, local_env):
    """ Evaluate the environment. If the environment is stochastic, every evaluation will
    return a different value of reward. The best individual is that one that survives on average
    across a lot of different generations, which is, the strongest-average-one. """

    # Reset conditions #
    local_env.reset()
    fitness = 0
    done_flag_ = False

    # Slice the individual array into agents actions #
    # t0 -> [1,2,1,1]
    # t1 -> [0,1,3,6]
    # ...
    # tN -> [2,7,1,7]

    action_array_ = np.asarray(np.split(ind, n_agents)).T

    # Select first actions ...
    actions_indexs_ = np.zeros(n_agents).astype(int) # The indexes of the actions of the individual for every agent #

    # Check the if actions are valid #
    dict_actions_ = {i: action_array_[a,i] for i, a in zip(local_env.agents.keys(), actions_indexs_)}
    safe_action_mask_ = np.asarray(local_env.check_actions(dict_actions_))
    # If not valid, iterate untill they are (BUT ONLY FOR THOSE WICH ARE INVALID)#
    while not all(safe_action_mask_):
        # Move the index if the agent action is not valid #
        actions_indexs_ = actions_indexs_ + safe_action_mask_.astype(int)
        dict_actions_ = {i: action_array_[a,i] for i, a in zip(local_env.agents.keys(), actions_indexs_)}
        safe_action_mask_ = np.asarray(local_env.check_actions(dict_actions_))

    # If the initial action is valid, begin to evaluate #
    while not done_flag_:

        _, reward_dict_, dones_dict_, _ = local_env.step(dict_actions_)

        # Again, check if the actions are still valid #
        safe_action_mask_ = np.asarray(local_env.check_actions(dict_actions_))
        # If not valid, iterate untill they are (BUT ONLY FOR THOSE WICH ARE INVALID)#
        while not all(safe_action_mask_):
            # Move the index if the agent action is not valid #
            actions_indexs_ = actions_indexs_ + safe_action_mask_.astype(int)
            dict_actions_ = {i: action_array_[a,i] for i, a in zip(local_env.agents.keys(), actions_indexs_)}
            safe_action_mask_ = np.asarray(local_env.check_actions(dict_actions_))

        # Accumulate the reward into the fitness #
        fitness += np.sum(list(reward_dict_.values()))
        # Check if done #
        done_flag_ = any(list(dones_dict_))


    return fitness,


def cxTwoPointCopy(ind1, ind2):

    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


toolbox.register("evaluate", evalEnv, local_env=env)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)


pool = multiprocessing.Pool()
toolbox.register("map", pool.map)


def optimize(local_env, save=False):


    random.seed(64)

    pop = toolbox.population(n=1000)

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
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=50, stats=stats, halloffame=hof)

    if save:

        with open(f"ga_simple_optimization_result_{datetime.datetime.now().strftime('%Y_%m_%d-%H:%M_%S')}.txt", "w") as solution_file:

            solution_file.write("Optimization result for the GA\n")
            solution_file.write("---------------------------------\n")
            solution_file.write(f"Starting positions: {local_env.initial_positions}\n")
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

    # Create the environment to optimize #
    my_env = SynchronousMultiAgentIGEnvironment(env_config=my_env_config)

    # Run for T times #
    for run in range(10):

        print("Run ", run)

        done_flag, t = False, 0

        R = 0

        # Initial reset #
        my_env.reset()

        """ OPTIMIZE THE SCENARIO """
        best = np.asarray(optimize(my_env))


        # Evaluate the metrics of the solutions #
        action_array = np.asarray(np.split(best, n_agents)).T
        actions_indexs = np.zeros(n_agents)  # The indexes of the actions of the individual for every agent #
        dict_actions = {i: action_array[a] for i, a in zip(my_env.agents.keys(), actions_indexs)}
        safe_action_mask = np.asarray(my_env.check_actions(dict_actions))

        # If not valid, iterate untill they are (BUT ONLY FOR THOSE WICH ARE INVALID)#
        while any(safe_action_mask):
            # Move the index if the agent action is not valid #
            actions_indexs = actions_indexs + safe_action_mask.astype(int)
            dict_actions = {i: action_array[a,i] for i, a in zip(my_env.agents.keys(), actions_indexs)}
            safe_action_mask = np.asarray(my_env.check_actions(dict_actions))

        # If the initial action is valid, begin to evaluate #
        while not done_flag:

            _, reward_dict, dones_dict, info = my_env.step(dict_actions)

            # Again, check if the actions are still valid #
            safe_action_mask = np.asarray(my_env.check_actions(dict_actions))
            # If not valid, iterate untill they are (BUT ONLY FOR THOSE WICH ARE INVALID)#
            while any(safe_action_mask):
                # Move the index if the agent action is not valid #
                actions_indexs = actions_indexs + safe_action_mask.astype(int)
                dict_actions = {i: action_array[a,i] for i, a in zip(my_env.agents.keys(), actions_indexs)}
                safe_action_mask = np.asarray(my_env.check_actions(dict_actions))

            # Accumulate the reward into the fitness #
            R += np.sum(list(reward_dict.values()))
            # Check if done #
            done_flag = any(list(dones_dict))


            metrics = [R,
                       np.mean(my_env.uncertainty),
                       my_env.max_sensed_value,
                       np.sum(info['collisions'])]

            evaluator.register_step(run_num=run, step=t, metrics=[*metrics])

            t += 1

evaluator.register_experiment()

pool.close()
