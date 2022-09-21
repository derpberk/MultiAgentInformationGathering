import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
import numpy as np
from Model.rllib_models import FullVisualQModel

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator

from ShekelGroundTruth import Shekel
from OilSpillEnvironment import OilSpill
from FireFront import WildfireSimulator


""" Initialize Ray """
ray.init()

""" Create a DQN config dictionary """
config = DEFAULT_CONFIG.copy()

""" Environment configuration"""
# The scenario
navigation_map = np.genfromtxt('./wesslinger_map.txt')
# Number of agents
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
	"num_workers": 1,
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

# Create the environment and set up the evaluation mode #
env = InformationGatheringEnv(env_config)
env.eval()

trainer = DQNTrainer(config=config)

trainer.restore('/Users/samuel/MultiAgentInformationGathering/Training/runs/checkpoint_000001/checkpoint-1')

# Create the evaluator and pass the metrics #
evaluator = MetricsDataCreator(metrics_names=['Mean Reward',
                                              'Average Uncertainty',
                                              'Mean regret',
                                              'Model Error'],
                               algorithm_name='DRL',
                               experiment_name='DeepReinforcementLearningResults')

paths = MetricsDataCreator(
							metrics_names=['vehicle', 'x', 'y'],
							algorithm_name='DRL',
							experiment_name='DeepReinforcementLearningResults_paths',
							directory='./')

gp = GaussianProcessRegressor(kernel=RBF(length_scale=15.0, length_scale_bounds=(2.0, 100.0)), alpha=0.001)

for run in range(20):

	# Reset the environment #
	t = 0
	obs = env.reset()
	done = {i: False for i in range(N)}
	done['__all__'] = False
	episode_reward = 0

	while not done['__all__']:

		# Compute action for every agent #
		action = {}
		for agent_id, agent_obs in obs.items():
			if not done[agent_id]:
				action[agent_id] = trainer.compute_action(agent_obs, policy_id='shared_policy', explore=False)

		# Send the computed action `a` to the env.
		obs, reward, done, info = env.step(action)
		env.render()

		# Save the reward #
		episode_reward += np.sum(list(reward.values()))

		# Let's test if there is much more improvement with a complimentary surrogate model
		gp.fit(env.measured_locations, env.measured_values)

		surr_mu, surr_unc = gp.predict(env.visitable_positions, return_std=True)
		real_mu = env.ground_truth.ground_truth_field[env.visitable_positions[:, 0], env.visitable_positions[:, 1]]

		mse = mean_squared_error(y_true=real_mu, y_pred=surr_mu, squared=False)

		metrics = [info['metrics']['accumulated_reward'],
		           info['metrics']['uncertainty'],
		           info['metrics']['instant_regret'],
		           mse
		           ]

		evaluator.register_step(run_num=run, step=t, metrics=[*metrics])
		for veh_id, veh in enumerate(env.fleet.vehicles):
			paths.register_step(run_num=run, step=t, metrics=[veh_id, veh.position[0], veh.position[1]])

		t += 1

	print(f"Episode done: Total reward = {episode_reward}")

# Register the metrics #
evaluator.register_experiment()
paths.register_experiment()
