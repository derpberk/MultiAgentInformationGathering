import ray
import numpy as np
from ray.rllib.algorithms.dqn.dqn import DQN

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

from Environment.IGEnvironments import InformationGatheringEnv
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator
from Training.TrainingConfigurations.ShekelConfiguration import create_configuration


""" Initialize Ray """
ray.init()

configuration = create_configuration()

# Create the environment and set up the evaluation mode #
env = InformationGatheringEnv(configuration.env_config)
env.eval()
N = env.number_of_agents
trainer = DQN(config=configuration)
trainer.restore(r'C:\Users\yane_sa\PycharmProjects\MultiAgentInformationGathering\Training\runs\DQN_2022-09-30_13-51-15\DQN_InformationGatheringEnv_30742_00000_0_2022-09-30_13-51-16\checkpoint_000001\checkpoint-1')

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
