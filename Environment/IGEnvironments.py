import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from OilSpillEnvironment import OilSpillEnv
from ShekelGroundTruth import Shekel
from Fleet import Fleet
from Vehicle import FleetState
import matplotlib.pyplot as plt


class InformationGatheringEnv(MultiAgentEnv):

	def __init__(self, env_config):

		super().__init__()

		self.env_config = env_config

		# Create a fleet of N vehicles #
		self.fleet = Fleet(fleet_configuration=env_config['fleet_configuration'])
		self.number_of_agents = env_config['fleet_configuration']['number_of_vehicles']
		self.agents_ids = list(range(self.number_of_agents))
		self._agent_ids = set(range(self.number_of_agents))

		# Define the observation space and the action space #
		self.observation_type = env_config['observation_type']

		if self.observation_type == 'visual':
			self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5, *env_config['navigation_map'].shape))
		elif self.observation_type == 'hybrid':
			# The hybrid state is its position [x,y] and 3 images [fleet positions, mean-model, uncertainty] #
			self.observation_space = gym.spaces.Dict({'visual_state': gym.spaces.Box(low=-1.0, high=1.0, shape=(3, *env_config['navigation_map'].shape)),
			                                          'odometry': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))})
		else:
			raise NotImplementedError('This observation type is not defined. Pleas choose between: visual / hybrid')

		if env_config['movement_type'] == 'DIRECTIONAL':
			self.action_space = gym.spaces.Discrete(env_config['number_of_actions'])
		elif env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
		else:
			raise NotImplementedError('This movement type is not defined. Pleas choose between: DIRECTIONAL / DIRECTIONAL+DISTANCE')

		self.measurements = None
		self.resetted = False
		self.rewards = {}
		self.states = {}
		self.infos = {}
		self.dones = {}

		# Reward related variables #
		""" Regression related values """
		self.random_benchmark = env_config['random_benchmark']
		if env_config['dynamic'] == 'OilSpillEnv':
			self.ground_truth = OilSpillEnv(self.env_config['navigation_map'], dt=1, flow=10, gamma=1, kc=1, kw=1, seed=self.env_config['seed'])
		elif env_config['dynamic'] == 'Shekel':
			self.ground_truth = Shekel(1 - self.env_config['navigation_map'], 1, max_number_of_peaks=4, is_bounded=True, seed=self.env_config['seed'])
		else:
			raise NotImplementedError("This benchmark is not implemented")

		self.visitable_positions = np.column_stack(np.where(env_config['navigation_map'] == 1))
		self.kernel = RBF(length_scale=env_config['kernel_length_scale'], length_scale_bounds=(0.1, 10.0))
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel, alpha=0.01, optimizer=None, n_restarts_optimizer=10)
		self.measured_values = None
		self.measured_locations = None
		self.mu = None
		self.uncertainty = None
		self.Sigma = None
		self.H_ = 0
		self.fig = None
		self.max_sensed_value = None
		self.regret = np.zeros((self.number_of_agents,))
		self.mse = None
		self.mse_ant = None
		self._eval = False
		self.uncertainty_component = 0
		self.regret_component = 0
		self.number_of_collisions = np.zeros(shape = (self.number_of_agents,))
		self.max_collisions = env_config['max_collisions']
		self._eval = env_config['eval_mode']

	def reset(self):
		""" Reset all the variables and the fleet """

		self.resetted = True

		self.dones = {i: False for i in range(self.number_of_agents)}
		self.dones['__all__'] = False

		# Reset the ground truth and set the value for
		self.ground_truth.reset(self.random_benchmark)
		self.update_vehicles_ground_truths()

		# Reset the model #
		self.measured_locations = None
		self.measured_values = None
		self.mu = None
		self.uncertainty = None
		self.Sigma = None
		self.uncertainty_component = 0
		self.regret_component = 0
		self.number_of_collisions[:] = 0
		self.mse_ant = None

		# Reset the vehicles and take the first measurements #
		self.measurements = self.fleet.reset()
		# Update the model with the initial values #
		self.mu, self.uncertainty, self.Sigma = self.update_model(new_measurements=self.measurements)
		# Compute the trace
		self.H_ = np.trace(self.Sigma)
		# Update the state
		self.states = self.update_states()

		return self.states

	def eval(self):
		print("Environment in eval mode!")
		self._eval = True

	def update_model(self, new_measurements, agents_ids = None):
		""" Fit the gaussian process using the measurements and return a new inferred map and its uncertainty """

		if agents_ids is None:
			agents_ids = self.agents_ids
		else:
			new_measurements = [new_measurements[i] for i in agents_ids]

		# Append the new data #
		if self.measured_locations is None:
			self.measured_locations = np.asarray([meas['position'] for meas in new_measurements])
			self.measured_values = np.asarray([np.nanmean(meas['data']) for meas in new_measurements])
		else:
			self.measured_locations = np.vstack((self.measured_locations, np.asarray([meas['position'] for meas in new_measurements])))
			self.measured_values = np.hstack((self.measured_values, np.asarray([np.nanmean(meas['data']) for meas in new_measurements])))

		# Obtain the max value obtained by the fleet to compute the regret #
		self.max_sensed_value = self.measured_values.max()
		# Compute the regret for those agents #
		self.regret[agents_ids] = self.max_sensed_value - self.measured_values[-len(agents_ids):]

		self.GaussianProcess.fit(self.measured_locations, self.measured_values)


		# Compute the mean and the covariance matrix #
		mu, Sigma = self.GaussianProcess.predict(self.visitable_positions[:], return_cov=True)

		self.mse_ant = self.mse
		self.mse = mean_squared_error(y_true=self.ground_truth.ground_truth_field[self.visitable_positions[:, 0], self.visitable_positions[:, 1]],
		                              y_pred=mu,
		                              squared=False)

		uncertainty = np.sqrt(Sigma.diagonal())

		mu_map = np.copy(self.env_config['navigation_map'])
		mu_map[self.visitable_positions[:, 0],
		       self.visitable_positions[:, 1]] = mu

		uncertainty_map = np.copy(self.env_config['navigation_map'])
		uncertainty_map[self.visitable_positions[:, 0],
		                self.visitable_positions[:, 1]] = uncertainty

		return mu_map, uncertainty_map, Sigma

	def reward_function(self, collision_array, agents_ids = None):
		""" The reward function has the following terms:
			H -> the decrement of the uncertainty from the previous instant to the next one. Defines the exploratory reward.
			"""

		if agents_ids is None:
			agents_ids = self.agents_ids

		Tr = np.trace(self.Sigma)

		self.uncertainty_component = (self.H_ - Tr) / self.number_of_agents
		self.regret_component = 2 - np.clip(self.regret, 0.0, 1.0)

		if self.env_config['reward_type'] == 'uncertainty':
			reward = self.uncertainty_component * self.regret_component
		elif self.env_config['reward_type'] == 'regret':
			reward = self.regret_component - 1
		elif self.env_config['reward_type'] == 'error':
			reward = np.array([self.mse_ant-self.mse for _ in range(self.number_of_agents)])
		else:
			raise NotImplementedError("Invalid reward type")

		reward[collision_array] = -1.0

		self.H_ = Tr

		return {i: reward[i] for i in agents_ids}

	def update_states(self, agent_ids=None):
		""" Update the states """
		if agent_ids is None:
			agent_ids = self.agents_ids

		s = {i: self.individual_state(i) for i in agent_ids}

		return s

	def individual_state(self, agent_indx):
		""" Return the state of an individual agent """

		other_agents_map = np.zeros_like(self.env_config['navigation_map'])
		other_agents_ids = np.copy(self.agents_ids)
		other_agents_ids = np.delete(other_agents_ids, agent_indx)

		for agent_id in other_agents_ids:
			agent_position = self.fleet.vehicles[agent_id].position.astype(int)
			other_agents_map[agent_position[0], agent_position[1]] = 1.0

		if self.observation_type == 'visual':

			# First channel: navigation/obstacle map
			nav_map = np.copy(self.env_config['navigation_map'])

			# Second channel: self-position map
			position_map = np.zeros_like(self.env_config['navigation_map'])
			agent_position = self.fleet.vehicles[agent_indx].position.astype(int)
			position_map[agent_position[0], agent_position[1]] = 1.0

			return np.concatenate((nav_map[np.newaxis],
			                       position_map[np.newaxis],
			                       other_agents_map[np.newaxis],
			                       np.clip(self.mu[np.newaxis], a_min = -1.0, a_max = 1.0),
			                       np.clip(self.uncertainty[np.newaxis], a_min = -1.0, a_max= 1.0)))

		elif self.observation_type == 'hybrid':

			return {'visual_state': np.concatenate((self.mu[np.newaxis],
			                                        self.uncertainty[np.newaxis],
			                                        other_agents_map[np.newaxis])),
			        'odometry': self.fleet.vehicles[agent_indx].position/self.env_config['navigation_map'].shape}


	def update_vehicles_ground_truths(self):
		for vehicle in self.fleet.vehicles:
			vehicle.ground_truth = self.ground_truth.ground_truth_field

	def action_dict_to_targets(self, a_dict):
		""" Transform the actions of the dictionary to a dictionary with the target goals """

		target_agents = list(a_dict.keys())

		if self.env_config['movement_type'] == 'DIRECTIONAL':
			# Transform discrete actions to displacement
			angles = np.asarray([2 * np.pi * action / self.env_config['number_of_actions'] for action in a_dict.values()])
			# Transform the displacement to positions #
			target_positions = self.fleet.get_positions()[target_agents] + self.env_config['measurement_distance'] * np.column_stack(
				(np.cos(angles), np.sin(angles)))

		elif self.env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			# Transform the continuous actions into displacement #
			actions = np.array([action for action in a_dict.values()])
			angles = np.pi * actions[:, 0]
			distances = self.linear_min_max(self.env_config['min_measurement_distance'], self.env_config['max_measurement_distance'], -1, 1,
			                                actions[:, 1])
			target_positions = self.fleet.get_positions()[target_agents] + distances.reshape(-1,1) * np.column_stack((np.cos(angles), np.sin(angles)))
		else:
			target_positions = None

		return target_positions

	@staticmethod
	def linear_min_max(y_min, y_max, x_min, x_max, x):
		""" Transform the input x into a line """
		return (y_max - y_min) / (x_max - x_min) * x + y_min

	def step(self, action_dict):
		""" Process the actions.
		 Note that you should only return those agent IDs in an observation dict,
		 for which you expect to receive actions in the next call to step()."""

		assert self.resetted, "You need to reset the environment first with env.reset()"

		# Compute the new target with the given actions #
		new_targets = self.action_dict_to_targets(action_dict)

		# Apply the target to the vehicles #
		for vehicle_id, target in zip(action_dict.keys(), new_targets):
			self.fleet.set_target_position(vehicle_id, target)

		# Step until any vehicle has completed (or failed its goal) #
		new_measurements = None

		if self.env_config['movement_type'] == 'DIRECTIONAL':
			# Update until all vehicles have arrived to their goals #
			_, new_measurements = self.fleet.update_syncronously()
		elif self.env_config['movement_type'] == 'DIRECTIONAL_DISTANCE':
			# Update until at least one vehicle has arrived to their goals#
			_, new_measurements = self.fleet.update_asyncronously()

		# Retrieve the ids of the agents that must return a state and a reward #
		ready_agents_ids = [i for i, veh_state in enumerate(self.fleet.fleet_state) if veh_state in [FleetState.WAITING_FOR_ACTION, FleetState.COLLIDED, FleetState.LAST_ACTION]]

		# Compute those agents that collided and add 1 collision to their counter #
		collision_array = [s == FleetState.COLLIDED for s in self.fleet.fleet_state]
		# Accumulate every agent collision #
		self.number_of_collisions += np.array(collision_array).astype(int)
		# For every agent, change its state to final if its number of collisions is equal to the maximum number of collisions #
		for agent_id in self.agents_ids:
			if self.number_of_collisions[agent_id] >= self.max_collisions:
				self.fleet.set_state(agent_id, FleetState.LAST_ACTION)

		done_agents_vals = [self.fleet.fleet_state[agents_id] == FleetState.LAST_ACTION for agents_id in ready_agents_ids]

		# Update the model
		self.mu, self.uncertainty, self.Sigma = self.update_model(new_measurements=new_measurements, agents_ids=ready_agents_ids)

		# Compute the rewards for those finished agents #
		self.rewards = self.reward_function(collision_array=collision_array, agents_ids=ready_agents_ids)
		self.number_of_collisions += np.array([int(collision) for collision in collision_array])

		# Compute the states for those same agents #
		self.states = self.update_states(agent_ids=ready_agents_ids)

		# Info is useless by the moment
		self.infos = {i: {'Collisions': self.number_of_collisions[i]} for i in ready_agents_ids}

		# Update the ground truth state and pass the field to agents #
		self.ground_truth.step()
		self.update_vehicles_ground_truths()

		# Compute if the agents have finished #
		self.dones = {i: val for i, val in zip(ready_agents_ids, done_agents_vals)}
		self.dones['__all__'] = all([veh_state in [FleetState.LAST_ACTION, FleetState.FINISHED] for veh_state in self.fleet.fleet_state])

		return self.states, self.rewards, self.dones, self.infos

	def render(self, mode='human'):


		if self.fig is None:

			plt.ion()

			self.fig, self.axs = plt.subplots(1, 5)

			self.axs[0].set_title('Navigation map')
			self.s0 = self.axs[0].imshow(self.env_config['navigation_map'], cmap='gray')

			self.axs[1].set_title('Fleet positions')
			if self.observation_type == 'visual':
				self.s1 = self.axs[1].imshow(np.sum([self.individual_state(i)[1] for i in range(self.number_of_agents)], axis=0), cmap='gray')
			elif self.observation_type == 'hybrid':
				self.s1 = self.axs[1].imshow(np.sum([self.individual_state(i)['visual_state'][2] for i in range(self.number_of_agents)], axis=0), cmap='gray')
			else:
				raise NotImplementedError('Cannot render with an invalid observation type.')

			self.axs[2].set_title('Estimated model')
			self.s2 = self.axs[2].imshow(self.mu, cmap='jet', vmin=0.0, vmax=1.0)

			self.axs[3].set_title('Uncertainty')
			self.s3 = self.axs[3].imshow(self.uncertainty, cmap='gray', vmin=0.0, vmax=1.0)

			self.axs[4].set_title('Ground truth')
			self.s4 = self.axs[4].imshow(self.ground_truth.ground_truth_field, cmap='jet', vmin=0.0, vmax=1.0)

		else:

			if self.observation_type == 'visual':
				self.s1.set_data(np.sum([self.individual_state(i)[1] for i in range(self.number_of_agents)], axis=0))
			elif self.observation_type == 'hybrid':
				self.s1.set_data(np.sum([self.individual_state(i)['visual_state'][2] for i in range(self.number_of_agents)], axis=0))

			self.s2.set_data(self.mu)
			self.s3.set_data(self.uncertainty)
			self.s4.set_data(self.ground_truth.ground_truth_field)

			self.fig.canvas.draw()

		plt.pause(0.2)
		plt.tight_layout(pad=0.5)
		plt.show()

	def get_action_mask(self, ind):
		""" Get the invalid action mask for a certain vehicle """

		angles = 2 * np.pi * np.arange(0, self.env_config['number_of_actions'])/self.env_config['number_of_actions']
		possible_points = self.fleet.vehicles[ind].position + self.env_config['measurement_distance'] * np.column_stack((np.cos(angles), np.sin(angles)))

		return np.asarray(list(map(self.fleet.vehicles[ind].is_the_position_valid, possible_points)))


if __name__ == '__main__':

	import matplotlib.pyplot as plt

	navigation_map = np.genfromtxt('./wesslinger_map.txt')

	N = 4

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
		'dynamic': 'Shekel',
		'min_measurement_distance': 5,
		'max_measurement_distance': 10,
		'measurement_distance': 3,
		'number_of_actions': 8,
		'kernel_length_scale': 2,
		'random_benchmark': True,
		'observation_type': 'visual',
		'max_collisions': 10,
		'eval_mode': True,
		'seed': 10,
		'reward_type': 'error'
	}

	# Create the environment #
	env = InformationGatheringEnv(env_config=env_config)

	state = env.reset()

	dones = {i: False for i in range(N)}
	dones['__all__'] = False

	H = []
	reg = []
	i_reg = []
	comb = []

	actions = {}
	for i in range(N):
		mask = env.get_action_mask(i)
		actions[i] = np.random.choice(np.arange(env.env_config['number_of_actions']), p=mask.astype(int)/np.sum(mask))

	while not dones['__all__']:

		print(""" ############################################################### """)
		print("Action: ", actions)

		states, rewards, dones, infos = env.step({i: actions[i] for i in dones.keys() if (not dones[i]) and i != '__all__'})

		for i in range(N):
			mask = env.get_action_mask(i)
			if not mask[actions[i]]:
				actions[i] = np.random.choice(np.arange(env.env_config['number_of_actions']), p=mask.astype(int)/np.sum(mask))


		H.append(np.mean(env.uncertainty_component))
		reg.append(np.mean(env.regret))
		i_reg.append(list(env.regret))
		comb.append(list(env.uncertainty_component * env.regret))

		print("States: ", states.keys())
		print("Rewards: ", rewards)
		print("Dones: ", dones)
		print("Info: ", infos)
		env.render()


	print("Finished!")
	plt.show(block=True)

	with plt.style.context('seaborn-whitegrid'):
		_, ax = plt.subplots(2, 1, sharex=True)

		ax[0].set_title('Uncertainty')
		ax[0].plot(H, 'k--', label = 'H')
		ax[0].plot(comb)
		ax[0].legend()
		ax[1].set_title('Regret')
		ax[1].plot(reg, 'k--')
		ax[1].plot(i_reg)

	plt.show(block=True)

