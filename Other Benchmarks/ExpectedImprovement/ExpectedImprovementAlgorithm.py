""" Algorithm that chooses next acquisition points
using the expected improvement reward in a BO style. """

import numpy as np
from Environment.InformationGatheringEnvironments import SynchronousMultiAgentIGEnvironment
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from skopt.acquisition import gaussian_ei
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

navigation_map = np.genfromtxt('../../Environment/wesslinger_map.txt')
n_agents = 3

agent_config = {'navigation_map': navigation_map,
                'mask_size': (1, 1),
                'initial_position': None,
                'speed': 1,
                'max_illegal_movements': 100,
                'max_distance': 200,
                'dt': 1}


my_env_config = {'number_of_agents':n_agents,
                 'number_of_actions': 8,
                 'kernel_length_scale': 2,
                 'agent_config': agent_config,
                 'navigation_map': navigation_map,
                 'meas_distance': 2,
                 'initial_positions': np.array([[21,14],[30,16],[36,41]]),
                 'max_meas_distance': 1,
                 'min_meas_distance': 1,
                 'dynamic': 'Shekel',
                 }


np.random.seed(0)

# Create the environment to optimize #
env = SynchronousMultiAgentIGEnvironment(env_config=my_env_config)

# Create the evaluator and pass the metrics #
evaluator = MetricsDataCreator(metrics_names=['Mean Reward', 'Uncertainty', 'Max regret', 'RMSE', 'Collisions'], algorithm_name='Greedy ExpImp', experiment_name='ExpectedImprovementResults')

def reverse_action(a):

    r_a = (a + my_env_config['number_of_actions']//2) % my_env_config['number_of_actions']

    return r_a

def predict_best_action(ei_map, last_actions):
    """ Compute the actions to move towards the maximal ei_map position """

    angles = np.asarray(list(map(env.action2angle, np.arange(0,my_env_config['number_of_actions']))))
    predicted_displacement = my_env_config['meas_distance'] * 1 *  np.asarray([np.cos(angles), np.sin(angles)]).T
    action_set = []

    if last_actions is None:
        last_actions = {idx : None for idx in env.agents.keys()}


    for pos, last_action, agent in zip(env.get_agents_positions(), last_actions.values(), env.agents.values()):

        # Compute the EI in the computed positions #
        next_observatory_positions = np.clip(pos + predicted_displacement, 0, np.asarray(navigation_map.shape)-1)

        values = ei_map[next_observatory_positions[:,0].astype(int), next_observatory_positions[:,1].astype(int)]

        # Discard the reversed last action #
        if last_action is not None:
            values[reverse_action(last_action)] = 0

        # Discard those values out of reach #
        illegal_mask = np.logical_not(env.get_valid_action_mask(agent_id=agent.agent_id))
        values[illegal_mask] = 0

        action_set.append(np.argmax(values))

    return {i:a for i,a in zip(env.agents.keys(), action_set)}


d = None

env.eval = True

# Evaluate for 10 scenarios #
for run in range(100):

    # Reset flags
    t, R, done_flag = 0, 0, False

    # Initial reset #
    env.reset()

    expected_improvement_map = np.copy(navigation_map)

    _dict_actions = None

    print("Run Nº ", run)

    while not done_flag:

        # ---- Optimization routine ---- #
        # 1. Compute the Expected Improvement map #
        expected_improvement_values = gaussian_ei(env.visitable_positions, y_opt=0, model=env.GaussianProcess, xi=1)
        expected_improvement_map[env.visitable_positions[:,0], env.visitable_positions[:,1]] = expected_improvement_values
        # 2. Compute the actions
        dict_actions = predict_best_action(expected_improvement_map, _dict_actions)

        _dict_actions = dict_actions

        _, reward_dict, dones_dict, info = env.step(dict_actions)

        # Accumulate the reward into the fitness #
        R += np.sum(list(reward_dict.values()))
        # Check if done #
        done_flag = any(list(dones_dict.values()))

        metrics = [R, np.mean(env.uncertainty), env.regret.mean(), env.mse, np.sum(info['collisions'])]

        evaluator.register_step(run_num=run, step=t, metrics=[*metrics])

        t += 1

        env.render()

        """
        if d is None:
            fig, axs = plt.subplots(1,3, figsize=(20,10))
            d = axs[0].imshow(expected_improvement_map, cmap='jet', vmin=0, vmax=1)
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(d, cax=cax, orientation='vertical')
            positions = env.get_agents_positions()
            d1, = axs[0].plot(positions[:,1].astype(int), positions[:,0].astype(int), 'ok', linewidth=5, markersize=10)
            d2 = axs[1].imshow(env.mu, cmap='jet', vmin=0, vmax=1)
            d3 = axs[2].imshow(env.ground_truth.ground_truth_field, cmap='jet', vmin=0, vmax=1)
        else:
            d.set_data(expected_improvement_map)
            positions = env.get_agents_positions()
            d1.set_data(positions[:,1], positions[:,0])
            d2.set_data(env.mu)
            d3.set_data(env.ground_truth.ground_truth_field)

        
        fig.canvas.draw()
        """

        plt.pause(0.01)



evaluator.register_experiment()