import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from Environment.IGEnvironments import InformationGatheringEnv
import numpy as np
from Model.rllib_models import FullVisualQModel

# Initialize Ray #
ray.init()

""" Create a DQN config dictionary """
config = DEFAULT_CONFIG.copy()

""" Environment configuration"""
navigation_map = np.genfromtxt('../Environment/wesslinger_map.txt')
N = 4

N = 4
same_evaluation_scenario = True
env_config = {
		'fleet_configuration': {
			'vehicle_configuration': {
				'dt': 0.5,
				'navigation_map': navigation_map,
				'target_threshold': 0.5,
				'ground_truth': np.random.rand(50,50),
				'measurement_size': np.array([2, 2]),
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
		'dynamic': 'OilSpillEnv',
		'min_measurement_distance': 5,
		'max_measurement_distance': 10,
		'measurement_distance': 3,
		'number_of_actions': 8,
		'kernel_length_scale': 2,
		'random_benchmark': False,
	}

""" Set the configuration for the training """

config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": InformationGatheringEnv,
    "env_config": env_config,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        'framestack': False,
        'custom_model': FullVisualQModel,
        "fcnet_hiddens": [64, 64],
        "no_final_linear": True,
    },

    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
	"num_workers": 1,
    "evaluation_num_workers": 0,
    "num_gpus": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
    "train_batch_size": 64,

    "explore": True,
        "exploration_config": {
           # Exploration subclass by name or full path to module+class
           # (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
           "type": "EpsilonGreedy",
           # Parameters for the Exploration class' constructor:
           "initial_epsilon": 1.0,
           "final_epsilon": 0.05,
           "epsilon_timesteps": 1000000  # Timesteps over which to anneal epsilon.

        },
    "replay_buffer_config": {
        # Enable the new ReplayBuffer API.
        "_enable_replay_buffer_api": True,
        "type": "MultiAgentPrioritizedReplayBuffer",
        # Size of the replay buffer. Note that if async_updates is set,
        # then each worker will have a replay buffer of this size.
        "capacity": 50000,
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # The number of continuous environment steps to replay at once. This may
        # be set to greater than 1 to support recurrent models.
        "replay_sequence_length": 1,
         },

    "target_network_update_freq": 20000,
    # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": True,
    # Dense-layer setup for each the advantage branch and the value branch
    # in a dueling architecture.
    "hiddens": [64, 64],
    # Whether to use double dqn
    "double_q": True,
    # N-step Q learning
    "n_step": 1,
	"multiagent": {
        # We only have one policy (calling it "shared").
        # Class, obs/act-spaces, and config will be derived
        # automatically.
        "policies": {"shared_policy"},
        # Always use "shared" policy.
        "policy_mapping_fn": (
            lambda agent_id, episode, **kwargs: "shared_policy"
        ),
    },

}

# Create our RLlib Trainer.
tune.run(DQNTrainer, config=config)
