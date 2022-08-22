import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from Environment.IGEnvironments import InformationGatheringEnv
import numpy as np
from Model.rllib_models import FullVisualActorModel

import ray.rllib.utils.exploration
from typing import Dict, Tuple
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

""" Initialize Ray """
ray.init()

""" Create a DQN config dictionary """
config = DEFAULT_CONFIG.copy()

""" Environment configuration"""
navigation_map = np.genfromtxt('../Environment/wesslinger_map.txt')
N = 4

same_evaluation_scenario = False
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
		'movement_type': 'DIRECTIONAL_DISTANCE',
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

eval_env_config = env_config.copy()

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
		'custom_model': FullVisualActorModel,
		"fcnet_activation": "relu",
		"no_final_linear": True,
	},

	# ===== RESOURCES ===== #
	# Number of parallel workers for sampling
	"num_workers": 2,
	# Number of workers for training
	"evaluation_num_workers": 1,
	"num_gpus": 0,
	"num_cpus_per_worker": 1,
	"num_envs_per_worker": 1,

	# ===== ROLLOUT SAMPLING ===== #
	# Size of the batches for training.
	"rollout_fragment_length": 30,

	# ===== EXPLORATION ===== #
	"explore": True,
	"exploration_config": {
		# The Exploration class to use. In the simplest case, this is the name
		# (str) of any class present in the `rllib.utils.exploration` package.
		# You can also provide the python class directly or the full location
		# of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
		# EpsilonGreedy").
		"type": "StochasticSampling",
		# Add constructor kwargs here (if any).
	},

	# ===== PPO LEARNING PARAMETERS ===== #
	"lambda": 0.95,
	"clip_param": 0.2,
	"train_batch_size": 1000,
	"use_critic": True,
	"use_gae": True,
	"kl_coeff": 1.0,
	"sgd_minibatch_size": 128,
	"num_sgd_iter": 30,
	"shuffle_sequences": True,
	"vf_loss_coeff": 1.0,
	"entropy_coeff": 0.0,
	"entropy_coeff_schedule": None,
	"vf_clip_param": 10.0,
	"kl_target": 0.01,

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

	"evaluation_interval": 10,
	"custom_eval_function": None,
	"metrics_num_episodes_for_smoothing": 100,  # Number of episodes to smooth over for computing metrics
}

""" Stop criterion """
stop_criterion = {
	"episodes_total": 20_000,
}

# Create our RLlib Trainer.
tune.run(PPOTrainer,
         config=config,
         stop=stop_criterion,
         local_dir='./runs',
         checkpoint_at_end=True,
         checkpoint_freq=1, )
