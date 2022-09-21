import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from Environment.IGEnvironments import InformationGatheringEnv
import numpy as np
from Model.rllib_models import FullVisualQModel

from ShekelGroundTruth import Shekel
from OilSpillEnvironment import OilSpill
from FireFront import WildfireSimulator

""" Initialize Ray """
ray.init()

""" Create a DQN config dictionary """
config = DEFAULT_CONFIG.copy()

""" Environment configuration"""
navigation_map = np.genfromtxt('../../Environment/wesslinger_map.txt')

# Number of agents
N = 4
# Set up the Ground Truth #
gt_config_file = Shekel.sim_config_template
gt_config_file['navigation_map'] = navigation_map

# S
env_config = InformationGatheringEnv.default_env_config
env_config['ground_truth'] = Shekel
env_config['ground_truth_config'] = gt_config_file
env_config['navigation_map'] = navigation_map
env_config['fleet_configuration']['number_of_vehicles'] = N
env_config['full_observable'] = True
env_config['reward_type'] = 'improvement'


""" Set the configuration for the training """

config = {
	# Environment (RLlib understands openAI gym registered strings).
	"env": InformationGatheringEnv,
	"env_config": env_config,
	"framework": "torch",

	# ===== MODEL ===== #
	# Tweak the default model provided automatically by RLlib,
	# given the environment's observation- and action spaces.
	"model": {
		'framestack': False,
		'custom_model': FullVisualQModel,
		"fcnet_hiddens": [64, 64],
		"no_final_linear": True,
		"conv_activation": "relu",
	},

	# ===== RESOURCES ===== #
	# Number of parallel workers for sampling
	"num_workers": 2,
	# Number of workers for training
	"evaluation_num_workers": 0,
	"num_gpus": 1,
	"num_cpus_per_worker": 1,
	"num_envs_per_worker": 1,
	# ===== ROLLOUT SAMPLING ===== #
	# Size of the batches for training.
	"train_batch_size": 64,
	"rollout_fragment_length": 4,
	# Update the training every training_intensity timesteps. #
	"training_intensity": 100,  # Relation between timesteps trained vs trainsteps sampled

	# ===== EXPLORATION ===== #
	"explore": True,
	"exploration_config": {
		# Exploration subclass by name or full path to module+class
		# (e.g. “ray.rllib.utils.exploration.epsilon_greedy.EpsilonGreedy”)
		"type": "EpsilonGreedy",
		# Parameters for the Exploration class' constructor:
		"initial_epsilon": 1.0,
		"final_epsilon": 0.05,
		"epsilon_timesteps": 3500000  # Timesteps over which to anneal epsilon.

	},

	# ===== DEEP Q LEARNING PARAMETERS ===== #

	# Buffer Replay #
	"replay_buffer_config": {
		# Enable the new ReplayBuffer API.
		"_enable_replay_buffer_api": True,
		"type": "MultiAgentPrioritizedReplayBuffer",
		# Size of the replay buffer. Note that if async_updates is set,
		# then each worker will have a replay buffer of this size.
		"capacity": 100000,
		"prioritized_replay_alpha": 0.6,
		# Beta parameter for sampling from prioritized replay buffer.
		"prioritized_replay_beta": 0.4,
		# Epsilon to add to the TD errors when updating priorities.
		"prioritized_replay_eps": 1e-6,
		# The number of continuous environment steps to replay at once. This may
		# be set to greater than 1 to support recurrent models.
		"replay_sequence_length": 1,
	},

	# Upate
	"target_network_update_freq": 30000,
	# === Model ===
	# Number of atoms for representing the distribution of return. When
	# this is greater than 1, distributional Q-learning is used.
	# the discrete supports are bounded by v_min and v_max
	"num_atoms": 41,
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
		"replay_mode": "independent",  # The experiences are independent of the agents and the steps #
		"count_steps_by": "env_steps",  # A timestep is a step in the environment, not one for agent that takes an action #
	},

	# ===== EVALUATION SETTINGS ===== #
	"evaluation_duration_unit": "episodes",
	"evaluation_duration": 10,

	"evaluation_config": {
		"render_env": False,
		"explore": False,
	},

	"evaluation_interval": 2000,
	"custom_eval_function": None,
	"metrics_num_episodes_for_smoothing": 100,  # Number of episodes to smooth over for computing metrics

}

""" Stop criterion """
stop_criterion = {
	"episodes_total": 100_000,
}

# Create our RLlib Trainer.
tune.run(DQNTrainer,
         config=config,
         stop=stop_criterion,
         local_dir='./runs',
         checkpoint_at_end=True,
         checkpoint_freq=100,)
