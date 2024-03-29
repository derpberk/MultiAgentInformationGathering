import sys
sys.path.append('.')
import gym
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from Fleet import Fleet
from Vehicle import FleetState
from typing import Union
from random import shuffle
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


# noinspection GrazieInspection
class InformationGatheringEnv:
	default_env_config = {
		'fleet_configuration': {
			'vehicle_configuration': {
				'dt': 0.05,
				'navigation_map': None,
				'target_threshold': 0.2,
				'ground_truth': None,
				'measurement_size': np.array([0, 0]),
				'max_travel_distance': 50,
			},
			'number_of_vehicles': 4,
			'initial_positions': np.array([[15, 19],
										   [13, 19],
										   [18, 19],
										   [15, 22]])
		},
		'movement_type': 'DIRECTIONAL',
		'navigation_map': np.ones((50, 50)),
		'min_measurement_distance': 3,
		'max_measurement_distance': 6,
		'measurement_distance': 3,
		'number_of_actions': 8,
		'kernel_length_scale': (3.5, 3.5, 50),
		'kernel_length_scale_bounds': ((0.1, 30), (0.1, 30), (0.001, 100)),
		'kernel_type': 'RBF',
		'random_benchmark': True,
		'observation_type': 'visual',
		'max_collisions': 10,
		'eval_mode': False,
		'seed': 10,
		'reward_type': 'improvement',
		'dynamic': False,
		'ground_truth': None,
		'ground_truth_config': None,
		'temporal': False,
		'full_observable': False,
	}

	def __init__(self, env_config: dict):

		super().__init__()

		# Environment configuration dictionary
		self.env_config = env_config
		self.env_config['fleet_configuration']['navigation_map'] = self.env_config['navigation_map']
		if self.env_config['fleet_configuration']['vehicle_configuration']['navigation_map'] is None:
			self.env_config['fleet_configuration']['vehicle_configuration']['navigation_map'] = self.env_config[
				'navigation_map']

		# Create a fleet of N vehicles #
		self.fleet = Fleet(fleet_configuration=self.env_config['fleet_configuration'])
		# Save the number of agents #
		self.number_of_agents = self.env_config['fleet_configuration']['number_of_vehicles']
		# Save the agents ids #
		self.agents_ids = list(range(self.number_of_agents))
		# Create a set of agents IDs - This is required for the RLLIB MA Environment class #
		self._agent_ids = set(range(self.number_of_agents))

		# Define the observation space and the action space #
		self.observation_type = self.env_config['observation_type']
		self.full_observable = self.env_config['full_observable']

		number_of_channels = 4 if self.env_config['reward_type'] == 'uncertainty' else 5
		if self.observation_type == 'visual':
			# The visual observation space is defined as 5 images:
			# - The navigation map
			# - The observer position on the map
			# - The other vehicles positions on the map
			# - The mu of the model
			# - The uncertainty of the model
			self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(
				number_of_channels, *self.env_config['navigation_map'].shape))
		elif self.observation_type == 'hybrid':
			# The hybrid state is its position [x,y] and 3 images [fleet positions, mean-model, uncertainty] #
			self.observation_space = gym.spaces.Dict(
				{'visual_state': gym.spaces.Box(low=-1.0, high=1.0, shape=(
					number_of_channels - 1, *self.env_config['navigation_map'].shape)),
				 'odometry': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))})
		else:
			raise NotImplementedError('This observation type is not defined. Pleas choose between: visual / hybrid')

		# Define the type of the action #
		if env_config['movement_type'] == 'DIRECTIONAL':
			# The action space is a discrete action space with number_of_actions actions:
			self.action_space = gym.spaces.Discrete(self.env_config['number_of_actions'])
		elif env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			# This action is defined with the direction and the distance to move (both normalized between -1 and 1) #
			self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
		else:
			raise NotImplementedError(
				'This movement type is not defined. Pleas choose between: DIRECTIONAL / DIRECTIONAL+DISTANCE')

		self.measurements = None  # Measurements of the environment (Usually a dictionary)
		self.resetted = False  # Flag to check if the environment has been resetted #

		# Gym Environment variables #
		self.rewards = {}
		self.states = {}
		self.infos = {}
		self.dones = {}

		# Ground truth - The task to solve #
		self.random_benchmark = env_config['random_benchmark']
		if self.env_config['ground_truth'] is not None:
			self.ground_truth = self.env_config['ground_truth'](self.env_config['ground_truth_config'])
		else:
			raise NotImplementedError("This benchmark is not implemented")

		# [N x 2] matrix with all the possible visitable positions #
		self.visitable_positions = np.column_stack(np.where(env_config['navigation_map'] == 1))

		# ---- Parameters of the model (Gaussian Process) ---- #
		# Kernel for model-conditioning #

		
		if env_config['temporal']:

			if env_config['kernel_type'] == 'RBF':
				self.kernel = RBF(length_scale=env_config['kernel_length_scale'],
								length_scale_bounds=env_config['kernel_length_scale_bounds'])
			elif env_config['kernel_type'] == 'matern':
				self.kernel = Matern(length_scale=env_config['kernel_length_scale'],
								length_scale_bounds=env_config['kernel_length_scale_bounds'])
		else:
			if env_config['kernel_type'] == 'RBF':
				self.kernel = RBF(length_scale=env_config['kernel_length_scale'][0],
								length_scale_bounds=env_config['kernel_length_scale_bounds'][0][0])
			elif env_config['kernel_type'] == 'matern':
				self.kernel = Matern(length_scale=env_config['kernel_length_scale'][0],
								length_scale_bounds=env_config['kernel_length_scale_bounds'][0][0])

		# The Gaussian Process #
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel,
														alpha=0.001,
														optimizer=None,
														n_restarts_optimizer=10,)
		# The list of measured values (y in GP)
		self.measured_values = None
		# The list of measured positions (x in GP)
		self.measured_locations = None
		# For anisotropic temporal modelling
		self.measurements_time = None

		# The surrogate model #
		self.mu = None
		# The surrogate uncertainty #
		self.uncertainty = None

		# Array of the last measurements of every agent (to compute the reward) #
		self.last_measurement_values = np.array([0.0 for _ in range(self.number_of_agents)])

		# The contribution of every agent to the uncertainty decrement (to compute the reward)#
		self.intermediate_uncertainty_values = np.ones(shape=(len(self.visitable_positions),))
		self.intermediate_mu_values = np.zeros(shape=(len(self.visitable_positions),))
		self.individual_uncertainty_decrement = np.array([0.0 for _ in range(self.number_of_agents)])
		self.individual_mse = np.array([0.0 for _ in range(self.number_of_agents)])

		# The error of the model #
		self.mse = None
		self.mse_ant = None
		self.acc_reward = 0

		# For the collision computation #
		self.number_of_collisions = np.zeros(shape=(self.number_of_agents,))
		self.max_collisions = env_config['max_collisions']

		self._eval = env_config['eval_mode']
		self.fig = None

	def reset(self) -> dict:
		""" Reset all the variables and the fleet. This method must be called before the first step of the episode.
		It resets the fleet position, the measurements, the ground truth, the surrogate model and the uncertainty.

		:return:
			The initial observation of the environment in a dictionary of agents.

		"""

		self.resetted = True

		# Reset the dones and the done flag #
		self.dones = {i: False for i in range(self.number_of_agents)}
		self.dones['__all__'] = False

		# Reset the ground truth and update the ground truth (if the ground truth is dynamic) #
		self.ground_truth.reset(self.random_benchmark)
		self.update_vehicles_ground_truths()

		# Reset the model parameters#
		self.measured_locations = None
		self.measured_values = None
		self.mu = np.zeros_like(self.env_config['navigation_map'])
		self.uncertainty = np.zeros_like(self.env_config['navigation_map'])
		self.number_of_collisions[:] = 0
		self.mse_ant = 0.0

		# Reset the fleet and take the first measurements #
		self.measurements = self.fleet.reset()

		# Update the model with the initial values #
		self.last_measurement_values = np.array([0.0 for _ in range(self.number_of_agents)])
		self.intermediate_uncertainty_values = np.ones(shape=(len(self.visitable_positions),))
		self.intermediate_mu_values = np.ones(shape=(len(self.visitable_positions),))
		self.individual_kl_divergence = np.array([0.0 for _ in range(self.number_of_agents)])
		self.individual_uncertainty_decrement = np.array([0.0 for _ in range(self.number_of_agents)])
		self.individual_model_change = np.array([0.0 for _ in range(self.number_of_agents)])
		self.individual_mse = np.zeros(shape=(self.number_of_agents,))
		self.mu, self.uncertainty = self.update_model(new_measurements=self.measurements)
		self.acc_reward = 0

		self.uncertainty_0 = self.uncertainty.sum()
		self.kl0 = np.sum(0.5 * (1.0 + (self.mu[self.visitable_positions[:,0], self.visitable_positions[:,1]]) ** 2) / ((self.uncertainty[self.visitable_positions[:,0], self.visitable_positions[:,1]] ** 2) - 0.5))

		# Update the state
		self.states = self.process_states()

		return self.states

	def eval(self):
		""" Change the environment to evaluation mode. In this mode,
		computationally expensive operations are enabled for evaluation. """

		print("Environment in eval mode!")
		self._eval = True

	def update_model(self, new_measurements: dict, agents_ids=None):
		"""
		Fit the gaussian process using the new_measurements and return a new inferred map and its uncertainty.
		The process will update the gaussian model sequentially for each agent in *agents_ids*.


		:param new_measurements: The new measurements to fit the model with in a dictionary.
		:param agents_ids: The ids of those agents that generated the measurements. If None, all the agents will be updated.

		:return:
			- mu - The new inferred map mu.
			- sigma - The new inferred uncertainty sigma.

		"""

		if agents_ids is None:
			agents_ids = self.agents_ids
		else:
			new_measurements = [new_measurements[i] for i in agents_ids]

		shuffled_list = list(zip(agents_ids, range(len(agents_ids))))
		shuffle(shuffled_list)
		for agent_id, measurement_id in shuffled_list:
			# Sequentially update the model for each agent. #

			# Append the data to the list of measurement locations and values #
			if self.measured_locations is None:
				self.measured_locations = np.asarray([new_measurements[measurement_id]['position']])
				# We compute the mean of the measurement-image #
				self.measured_values = np.asarray([np.nanmean(new_measurements[measurement_id]['data'])])
				# If the mission is of a temporal-patrolling kind append also the time of the sample #
				if self.env_config['temporal']:
					self.measurements_time = np.atleast_2d(np.asarray([new_measurements[measurement_id]['time']]))

			else:
				self.measured_locations = np.vstack(
					(self.measured_locations, np.asarray([new_measurements[measurement_id]['position']])))
				self.measured_values = np.hstack(
					(self.measured_values, np.asarray([np.nanmean(new_measurements[measurement_id]['data'])])))
				# If the mission is of a temporal-patrolling kind append also the time of the sample #
				if self.env_config['temporal']:
					self.measurements_time = np.vstack(
						(self.measurements_time, np.asarray([new_measurements[measurement_id]['time']])))

			# Store this last measured value #
			self.last_measurement_values[agent_id] = self.measured_values[-1]

			# Fit the gaussian process #
			if self.env_config['temporal']:
				self.GaussianProcess.fit(np.hstack((self.measured_locations, self.measurements_time)),self.measured_values)
				# Compute the mean and the std values for all the visitable positions #
				mu_array, sigma_array = self.GaussianProcess.predict(np.hstack((self.visitable_positions[:], np.full((len(self.visitable_positions), 1), np.max(self.measurements_time)))),
																	 return_std=True)
			else:
				self.GaussianProcess.fit(self.measured_locations, self.measured_values)
				# Compute the mean and the std values for all the visitable positions #
				mu_array, sigma_array = self.GaussianProcess.predict(self.visitable_positions[:], return_std=True)

			# Clip to 0 #

			mu_array = np.clip(mu_array, 0.0, 1.0)

			# Compute the new mean error #
			self.mse = np.mean(np.abs(self.ground_truth.ground_truth_field[self.visitable_positions[:, 0], self.visitable_positions[:, 1]] - mu_array))
			self.individual_mse[agent_id] = - self.mse + self.mse_ant
			self.mse_ant = self.mse

			# Compute the KL distance respect the latest obtained model #
			kl_div = np.abs(np.log(sigma_array/self.intermediate_uncertainty_values) + 0.5 * (self.intermediate_uncertainty_values**2 + (mu_array - self.intermediate_mu_values)**2)/(sigma_array**2) - 0.5)
			self.individual_kl_divergence[agent_id] = np.mean(np.tanh(kl_div))

			# Save the intermediate uncertainty/mean maps for uncertainty/mean credit assignment #
			self.individual_uncertainty_decrement[agent_id] = np.sum(self.intermediate_uncertainty_values - sigma_array)
			self.individual_model_change[agent_id] = np.sum(mu_array - self.mu[self.visitable_positions[:, 0], self.visitable_positions[:, 1]])
			self.intermediate_uncertainty_values = sigma_array.copy()
			self.intermediate_mu_values = mu_array.copy()

		# Conform the map of the surrogate model for the state #
		mu_map = np.copy(self.env_config['navigation_map'])
		mu_map[self.visitable_positions[:, 0],
			   self.visitable_positions[:, 1]] = mu_array

		uncertainty_map = np.copy(self.env_config['navigation_map'])
		uncertainty_map[self.visitable_positions[:, 0],
						self.visitable_positions[:, 1]] = sigma_array

		return mu_map, uncertainty_map

	def reward_function(self, collision_array: list, agents_ids=None) -> dict:
		""" The reward function is defined depending on the reward_type parameter
		1) 'uncertainty': the reward is merely the uncertainty decrement of each agent. This will serve for complete coverage
		2) 'regret': the reward is the decrement of uncertainty but weighted with the sampling value of each agent. This will serve for finding maxima
		3) 'error': the reward is the error between the ground truth and the inferred map. This is for characterization.

		:param collision_array: The collision array of the current state.
		:param agents_ids: The ids of those agents that are expecting the reward.

		:return:
			- reward: The reward for each agent in a dictionary.
		"""

		if agents_ids is None:
			agents_ids = self.agents_ids

		if self.env_config['reward_type'] == 'uncertainty':
			reward = 100 * self.individual_uncertainty_decrement / self.uncertainty_0
		elif self.env_config['reward_type'] == 'improvement':
			reward = 100 * self.individual_uncertainty_decrement * (1.0 + self.last_measurement_values) / self.uncertainty_0
		elif self.env_config['reward_type'] == 'improvement2':
			reward = 100 * self.individual_uncertainty_decrement * self.individual_model_change / self.uncertainty_0
		elif self.env_config['reward_type'] == 'kl':
			reward = 100 * self.individual_kl_divergence
		elif self.env_config['reward_type'] == 'error':
			reward = self.individual_mse.copy()
		else:
			raise NotImplementedError("Invalid reward type")

		# Penalize the agents that collided #
		reward[collision_array] = -1.0

		return {i: reward[i] for i in agents_ids}

	def process_states(self, agent_ids=None) -> dict:
		""" Render the states of the agents.

		:param agent_ids: The ids of those agents that are expecting the state.

		:return:
			- states: The state of the agents in a dictionary.

		"""

		if agent_ids is None:
			agent_ids = self.agents_ids

		s = {i: self.individual_state(i) for i in agent_ids}

		return s

	def individual_state(self, agent_indx: int) -> Union[np.ndarray, dict]:
		""" Return the state of an individual agent.

		:param agent_indx: The index of the agent.

		:return:
			- state: The state of the agent. Could be a matrix (visual) or a dictionary (hybrid).

		"""

		other_agents_map = np.zeros_like(self.env_config['navigation_map'])
		other_agents_ids = np.copy(self.agents_ids)
		other_agents_ids = np.delete(other_agents_ids, agent_indx)

		for agent_id in other_agents_ids:
			agent_position = self.fleet.vehicles[agent_id].position.astype(int)

			if self.fleet.fleet_state[agent_id] != FleetState.FINISHED:
				other_agents_map[agent_position[0], agent_position[1]] = 1.0

		# WARNING: Prepare the model observation depending on the full-visual condition #
		# This is only for comparing the performance with an idealistic scenario where we have all the model #
		mu = self.mu if self.full_observable is False else self.ground_truth.ground_truth_field

		if self.observation_type == 'visual':

			# First channel: navigation/obstacle map
			nav_map = np.copy(self.env_config['navigation_map'])

			# Second channel: self-position map
			position_map = np.zeros_like(self.env_config['navigation_map'])
			agent_position = self.fleet.vehicles[agent_indx].position.astype(int)

			if self.fleet.fleet_state[agent_indx] != FleetState.FINISHED:
				position_map[agent_position[0], agent_position[1]] = 1.0
			else:
				position_map[agent_position[0], agent_position[1]] = 0.0

			# Note that the mu and sigma maps are already normalized to [0, 1]
			if self.env_config['reward_type'] == 'uncertainty':
				return np.concatenate((nav_map[np.newaxis],
									   position_map[np.newaxis],
									   other_agents_map[np.newaxis],
									   np.clip(self.uncertainty[np.newaxis], a_min=-1.0, a_max=1.0)))
			else:
				return np.concatenate((nav_map[np.newaxis],
									   position_map[np.newaxis],
									   other_agents_map[np.newaxis],
									   np.clip(mu[np.newaxis], a_min=-1.0, a_max=1.0),
									   np.clip(self.uncertainty[np.newaxis], a_min=-1.0, a_max=1.0)))

		elif self.observation_type == 'hybrid':

			if self.env_config['reward_type'] == 'uncertainty':

				return {'visual_state': np.concatenate((np.clip(self.uncertainty[np.newaxis], a_min=-1.0, a_max=1.0),
														other_agents_map[np.newaxis])),
						'odometry': self.fleet.vehicles[agent_indx].position / self.env_config['navigation_map'].shape}
			else:
				return {'visual_state': np.concatenate((np.clip(mu[np.newaxis], a_min=-1.0, a_max=1.0),
														np.clip(self.uncertainty[np.newaxis], a_min=-1.0, a_max=1.0),
														other_agents_map[np.newaxis])),
						'odometry': self.fleet.vehicles[agent_indx].position / self.env_config['navigation_map'].shape}

	def update_vehicles_ground_truths(self):
		""" Setter to update the ground truth of the vehicles. """

		self.ground_truth.step()
		for vehicle in self.fleet.vehicles:
			vehicle.ground_truth = self.ground_truth.ground_truth_field

	def action_dict_to_targets(self, a_dict: dict) -> np.ndarray:
		""" Transform the actions of the dictionary to a dictionary with the target goals.

		:param a_dict: The dictionary of actions. Every key is an agent_id, and the values corresponds to the action.

		:return:
			- targets: An array that contains the target positions for every vehicle.

		"""

		target_agents = list(a_dict.keys())

		if self.env_config['movement_type'] == 'DIRECTIONAL':
			# Transform discrete actions to displacement
			angles = np.asarray([2 * np.pi * action / self.env_config['number_of_actions'] for action in a_dict.values()])
			# Transform the displacement to positions #
			target_positions = self.fleet.get_positions()[target_agents] + self.env_config['measurement_distance'] * np.column_stack((np.cos(angles), np.sin(angles)))

		elif self.env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			# Transform the continuous actions into displacement #
			acts = np.array([action for action in a_dict.values()])
			angles = np.pi * acts[:, 0]
			distances = self.linear_min_max(self.env_config['min_measurement_distance'],
											self.env_config['max_measurement_distance'], -1, 1,
											acts[:, 1])
			target_positions = self.fleet.get_positions()[target_agents] + distances.reshape(-1, 1) * np.column_stack((np.cos(angles), np.sin(angles)))

		else:
			raise NotImplementedError("Invalid movement type")

		return target_positions

	@staticmethod
	def linear_min_max(y_min, y_max, x_min, x_max, x):
		""" Transform the input x into a line """
		return (y_max - y_min) / (x_max - x_min) * x + y_min

	def compute_metrics(self, mission_type='MaximaSearch'):
		""" Here we compute every metric related to the metrics of the mission.
		Depending on the mode, one kind of metrics are calculated.

		:param mission_type: The type of mission we are performing. It will define what metrics are significant to the mission.

		:return: A dictionary with the metrics.
		"""

		metric_dict = {'error': self.mse,
					   'accumulated_reward': self.acc_reward,
					   'uncertainty': self.uncertainty[self.visitable_positions[:, 0], self.visitable_positions[:, 1]].mean(),
					   'model_change': self.individual_model_change.mean(),
					   'kl_divergence': self.individual_kl_divergence.mean()
					   }

		if mission_type == 'MaximaSearch':
			pass
			# TODO: Implement the metrics

		else:
			raise NotImplementedError("This is not a valid mission type:")

		return metric_dict

	def step(self, action_dict: dict):
		""" Process the actions. The action is processed for those waiting vehicles. The fleet is updated until one/all vehicles are ready.
		Every vehicle takes a measurement and the reward is processed

		:param action_dict: The dictionary of actions. Every key is an agent_id, and the values corresponds to the action.

		:return:
			- reward: The reward for each agent.
			- observation: The observation for each agent.
			- done: A boolean that indicates if the episode is finished.
			- info: A dictionary that contains additional information.

		"""

		assert self.resetted, "You need to reset the environment first with env.reset()"

		if action_dict == {}:
			return {}, {}, {}, {}

		# Compute the new target with the given actions #
		new_targets = self.action_dict_to_targets(action_dict)

		# Apply the new target to the vehicles #
		for vehicle_id, target in zip(action_dict.keys(), new_targets):
			self.fleet.set_target_position(vehicle_id, target)

		# Step until any vehicle has completed (or failed its goal) #
		new_measurements = None

		if self.env_config['dynamic']:
			self.update_vehicles_ground_truths()
		# TODO: Implement a method to update depending on the time that has passed since the last time

		if self.env_config['movement_type'] == 'DIRECTIONAL':
			# Update until ALL vehicles have arrived to their goals #
			_, new_measurements = self.fleet.update_syncronously()

		elif self.env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			# Update until AT LEAST ONE vehicle has arrived to their goals#
			_, new_measurements = self.fleet.update_asyncronously()

		# Retrieve the ids of the agents that must return a state and a reward #
		ready_agents_ids = [i for i, veh_state in enumerate(self.fleet.fleet_state) if
							veh_state in [FleetState.WAITING_FOR_ACTION, FleetState.COLLIDED, FleetState.LAST_ACTION]]

		# Compute those agents that collided and add 1 collision to their counter #
		collision_array = [s == FleetState.COLLIDED for s in self.fleet.fleet_state]
		# Accumulate every agent collision #
		self.number_of_collisions += np.array(collision_array).astype(int)
		# For every agent, change its state to final if its number of collisions is equal to the maximum number of collisions #
		for agent_id in self.agents_ids:
			if self.number_of_collisions[agent_id] >= self.max_collisions and self.fleet.fleet_state[
				agent_id] != FleetState.FINISHED:
				self.fleet.set_state(agent_id, FleetState.LAST_ACTION)

		done_agents_vals = [self.fleet.fleet_state[agents_id] == FleetState.LAST_ACTION for agents_id in
							ready_agents_ids]

		# Update the model
		self.mu, self.uncertainty = self.update_model(new_measurements=new_measurements, agents_ids=ready_agents_ids)

		# Compute the rewards for those finished agents #
		self.rewards = self.reward_function(collision_array=collision_array, agents_ids=ready_agents_ids)

		# Compute the states for those same agents #
		self.states = self.process_states(agent_ids=ready_agents_ids)

		# Info is useless by the moment
		self.infos = {i: {'Collisions': self.number_of_collisions[i]} for i in ready_agents_ids}

		# Compute if the agents have finished #
		self.dones = {i: val for i, val in zip(ready_agents_ids, done_agents_vals)}
		self.dones['__all__'] = all(
			[veh_state in [FleetState.LAST_ACTION, FleetState.FINISHED] for veh_state in self.fleet.fleet_state])

		# Accumulate the reward #
		self.acc_reward += np.asarray([val for val in self.rewards.values()]).sum()

		# Compute the metrics (if we are in eval mode) #
		if self._eval:
			self.infos['metrics'] = self.compute_metrics() if self._eval else {}

		return self.states, self.rewards, self.dones, self.infos

	def render(self, mode='human'):
		""" Render the environment. """

		if self.fig is None:

			plt.ion()

			self.fig, self.axs = plt.subplots(1, 5)

			self.axs[0].set_title('Navigation map')
			self.s0 = self.axs[0].imshow(self.env_config['navigation_map'], cmap='gray')

			self.axs[1].set_title('Fleet positions')
			if self.observation_type == 'visual':
				self.s1 = self.axs[1].imshow(np.sum([self.individual_state(i)[1] for i in range(self.number_of_agents)], axis=0), cmap='gray')
			elif self.observation_type == 'hybrid':
				self.s1 = self.axs[1].imshow(np.sum([self.individual_state(i)['visual_state'][2] for i in range(self.number_of_agents)], axis=0),cmap='gray')
			else:
				raise NotImplementedError('Cannot render with an invalid observation type.')

			self.axs[2].set_title('Estimated model')
			self.s2 = self.axs[2].imshow(self.mu, cmap='jet', vmin=0.0, vmax=1.0)

			self.axs[3].set_title('Uncertainty')
			unc = self.uncertainty.copy() * np.nan
			unc[self.visitable_positions[:, 0], self.visitable_positions[:, 1]] = self.uncertainty[
				self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
			self.s3 = self.axs[3].imshow(unc, cmap='gray', vmin=0, vmax=1)

			self.axs[4].set_title('Ground truth')
			self.s4 = self.axs[4].imshow(self.ground_truth.ground_truth_field, cmap='jet', vmin=0.0, vmax=1.0)
			self.s5, = self.axs[4].plot(self.measured_locations[:,1], self.measured_locations[:,0], 'wx', markersize=6)

		else:

			if self.observation_type == 'visual':
				self.s1.set_data(np.sum([self.individual_state(i)[1] for i in range(self.number_of_agents)], axis=0))
			elif self.observation_type == 'hybrid':
				self.s1.set_data(
					np.sum([self.individual_state(i)['visual_state'][2] for i in range(self.number_of_agents)], axis=0))

			self.s2.set_data(self.mu)
			unc = self.uncertainty.copy()*np.nan
			unc[self.visitable_positions[:, 0], self.visitable_positions[:, 1]] = self.uncertainty[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]

			self.s3.set_data(unc)
			self.s4.set_data(self.ground_truth.ground_truth_field)
			self.s5.set_data((self.measured_locations[:,1], self.measured_locations[:,0]))

			self.fig.canvas.draw()

		plt.pause(0.01)
		plt.tight_layout(pad=0.5)
		plt.show()

	def get_action_mask(self, ind: int) -> np.ndarray:
		""" Get the invalid action mask for a certain vehicle (*ind*)

		:param ind: The index of the vehicle

		:return:
			- mask: The invalid action mask for the vehicle, where true means that the action is valid.

		"""

		assert self.env_config[
				   'movement_type'] == 'DIRECTIONAL', 'This function is only valid for DIRECTIONAL movement.'

		angles = 2 * np.pi * np.arange(0, self.env_config['number_of_actions']) / self.env_config['number_of_actions']
		possible_points = self.fleet.vehicles[ind].position + 1.05 * self.env_config[
			'measurement_distance'] * np.column_stack(
			(np.cos(angles), np.sin(angles)))

		return np.asarray(list(map(self.fleet.vehicles[ind].is_the_position_valid, possible_points)))


if __name__ == '__main__':

	import matplotlib.pyplot as plt
	from ShekelGroundTruth import Shekel
	from OilSpillEnvironment import OilSpill
	from FireFront import WildfireSimulator

	# The scenario
	navigation_map = np.genfromtxt('Environment/SquaredMap.txt')
	# Number of agents
	N = 4
	# Set up the Ground Truth #
	gt_config_file = WildfireSimulator.sim_config_template
	gt_config_file['navigation_map'] = navigation_map

	env_config = InformationGatheringEnv.default_env_config
	env_config['ground_truth'] = WildfireSimulator
	env_config['ground_truth_config'] = gt_config_file
	env_config['navigation_map'] = navigation_map
	env_config['fleet_configuration']['number_of_vehicles'] = N
	env_config['fleet_configuration']['vehicle_configuration']['max_travel_distance'] = 1000
	env_config['kernel_length_scale'] = (3.5, 3.5, 50)
	env_config['kernel_length_scale_bounds'] = ((0.1, 30), (0.1, 30), (0.001, 100)),
	env_config['full_observable'] = False
	env_config['reward_type'] = 'kl'

	# Create the environment #
	env = InformationGatheringEnv(env_config=env_config)

	state = env.reset()

	dones = {i: False for i in range(N)}
	dones['__all__'] = False

	H = []
	mse = []
	rew = []
	colli = []

	actions = {i: env.action_space.sample() for i in range(N)}

	for i in range(N):
		mask = env.get_action_mask(i)
		if not mask[actions[i]]:
			actions[i] = np.random.choice(np.arange(env.env_config['number_of_actions']),
										  p=mask.astype(int) / np.sum(mask))

	while not dones['__all__']:

		print(""" ############################################################### """)
		print("Action: ", actions)

		states, rewards, dones, infos = env.step(
			{i: actions[i] for i in dones.keys() if (not dones[i]) and i != '__all__'})

		for i in range(N):
			mask = env.get_action_mask(i)
			if not mask[actions[i]]:
				actions[i] = np.random.choice(np.arange(env.env_config['number_of_actions']),
											  p=mask.astype(int) / np.sum(mask))

		H.append(env.uncertainty.mean())
		mse.append(env.mse)
		rew.append(np.mean(list(rewards.values())))
		colli.append(list(env.number_of_collisions))

		print("States: ", states.keys())
		print("Rewards: ", rewards)
		print("Dones: ", dones)
		print("Info: ", infos)
		print("Fleet state", env.fleet.fleet_state)

		env.render()

	print("Finished!")

	plt.show(block=True)

	with plt.style.context('seaborn-whitegrid'):
		_, ax = plt.subplots(4, 1, sharex='all')

		ax[0].set_ylabel('Uncertainty')
		ax[0].plot(H, '-', linewidth=2)
		ax[1].set_ylabel('error')
		ax[1].plot(mse, '-', linewidth=2)
		ax[2].set_ylabel('Reward')
		ax[2].plot(rew, '-', linewidth=2)
		ax[3].set_ylabel('Collisions')
		ax[3].plot(colli, '-', linewidth=2)

	plt.show(block=True)

	print(np.sum(rew))

	plt.scatter(rew, H)
	plt.show(block=True)

	plt.scatter(rew, mse)
	plt.show(block=True)
