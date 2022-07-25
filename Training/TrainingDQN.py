import ray
from ray import tune
import ray.rllib.agents.dqn as dqn
from Environment.IGEnvironments import InformationGatheringEnv
import numpy as np
from Model.rllib_models import FullVisualQModel

# Initialize Ray #
ray.init()

""" Create a DQN config dictionary """
config = dqn.DEFAULT_CONFIG.copy()

""" Environment configuration"""
navigation_map = np.genfromtxt('/Users/samuel/MultiAgentInformationGathering/Environment/wesslinger_map.txt')
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

""" Set the configuration for the training """
config["num_workers"] = 1
config["num_envs_per_worker"] = 1
config["num_gpus"] = 1
config["train_batch_size"] = 64
config["lr"] = 1e-4
config["env"] = InformationGatheringEnv
config["env_config"] = env_config
config["model"] = FullVisualQModel
config["rollout_fragment_length"] = 4,
config["framework"] = 'torch'
config['exploration_config'] = {'type': "EpsilonGreedy",
                                'initial_epsilon': 1.0,
                                'final_epsilon': 0.02,
                                'epsilon_timesteps': 10000}

config['multiagent'] = {'policies': {},
                        'policy_map_capacity': 100,
                        'policy_map_cache': None,
                        'policy_mapping_fn': None,
                        'policies_to_train': None,
                        'observation_fn': None,
                        'replay_mode': 'independent',
                        'count_steps_by': 'env_steps'},

config['double_q'] = True
config["replay_buffer_config"]= {
            "type": "PrioritizedReplayBuffer",
            # Size of the replay buffer. Note that if async_updates is set,
            # then each worker will have a replay buffer of this size.
            "capacity": 100_000,
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # The number of continuous environment steps to replay at once. This may
            # be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
        }

