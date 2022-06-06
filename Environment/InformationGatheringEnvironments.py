import gym
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt


class QuadcopterAgent(gym.Env):

	_action_types = ['CartesianDiscrete', 'AngleAndDistance']

	def __init__(self,
	             navigation_map: np.ndarray,
	             ground_truth_field,
	             mask_size,
	             max_distance,
	             max_illegal_movements,
	             action_type: dict,
	             initial_position = None):

		""" Quadcopter Agent Class:
		 Serves as an abstract class for the vehicle movement processing.
		 It defines the individual mesurement process.  """

		self.navigation_map = navigation_map

		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))
		self.initial_position = initial_position

		# A dictionary with the action parameters #
		self.action_type = action_type
		# Initial position #
		self.position = None
		# Initial measurement mask #
		self.measurement = None
		self.mask_size = mask_size

		# Other variables #
		self.illegal_movements = 0
		self.max_illegal_movements = max_illegal_movements
		self.max_distance = max_distance
		self.distance = 0
		self._ground_truth_field = ground_truth_field

	@property
	def ground_truth_field(self):
		return self._ground_truth_field
	
	@ground_truth_field.setter
	def ground_truth_field(self, new_value):
		self._ground_truth_field = new_value

	def step(self, action):
		""" Process the action given the action type """

		# A Castesian Discrete [N,S,E,W,NE,NW,SE,SW] movements with fixed distance
		if self.action_type['type'] == 'CartesianDiscrete':
			# Cardinal to radians
			angle = self.action_type['value']/ self.action_type['max'] * 2 * np.pi
			# Compute next attempted position #
			next_attempted_position = self.position + self.action_type['max_distance'] * np.asarray([np.cos(angle), np.sin(angle)])
			# Add the travelled distance
			self.distance += self.action_space['max_distance']

		# Angle and distance given as actions #
		elif self.action_type['type'] == 'AngleAndDistance':

			# Unpack
			angle, distance = self.action_type['value']
			# Compute next attempted position #
			next_attempted_position = self.position + distance * np.asarray([np.cos(angle), np.sin(angle)])
			self.distance += distance

		else:
			raise ValueError("Action type for quadcopter not correct. Choose a correct one")

		# Check if it is valid (without taking in account other agents)
		valid = self.check_movement_feasibility(next_attempted_position)

		if valid:
			# Update the position #
			self.position = next_attempted_position
			# Take a measurement #
			self.measurement = self.take_measurement()

		else:
			self.illegal_movements += 1

		done = self.illegal_movements >= self.max_illegal_movements or self.distance > self.max_distance

		return self.measurement, valid, done, {}

	def check_movement_feasibility(self, pos):
		""" Check if the movement is possible with the current navigation map.
			This method should be overriden when using the real deploy agent. """
		visitable = self.navigation_map[pos[0], pos[1]] == 1
		in_bounds = all(pos >= self.navigation_map.shape) and all(pos < 0)

		return visitable and in_bounds

	def take_measurement(self, pos = None):

		if pos is None:
			pos = self.position

		measurement_mask = self._ground_truth_field[pos[0] - self.mask_size[0]: pos[0] + self.mask_size[0] + 1,
		                                            pos[1] - self.mask_size[1]: pos[0] + self.mask_size[1] + 1]

		return measurement_mask

	def reset(self):
		""" Reset the positions, measuremente, and so on. """

		if self.initial_position is None:
			self.position = self.visitable_positions[np.random.randint(0, len(self.visitable_positions))]
		else:
			self.position = np.copy(self.initial_position)

		self.measurement = self.take_measurement()
		self.distance = 0
		self.illegal_movements = 0

	def render(self, mode='human'):
		raise NotImplementedError("This method is not implemented yet")


class MultiAgentOilSpillEnvironment(gym.Env):

	def __init__(self, env_config):
		super().__init__()

		# Create N agents #
		self.env_config = env_config
		self.agents = [QuadcopterAgent(env_config['agent_config']) for _ in range(env_config['number_of_agents'])]
		self._agent_ids = tuple([f'agent_{i}' for i in range(env_config['number_of_agents'])])
		self.dones = [False for _ in range(env_config['number_of_agents'])]
		self.observation_space = gym.spaces.Box(low=0.0, high=10000, shape=(4, *env_config['navigation_map'].shape))

		# Set the gym.Env mandatory parameters #
		if env_config['agent_config']['action_type']['type'] == 'CartesianDiscrete':
			self.action_space = gym.spaces.Discrete(env_config['agent_config']['action_type']['type'])
		elif env_config['agent_config']['action_type']['type'] == 'AngleAndDistance':
			self.action_space = gym.spaces.Discrete(env_config['agent_config']['action_type']['type'])
		else:
			raise ValueError("Action type for quadcopter not correct. Choose a correct one")

		self.resetted = False

		self.states, self.measurements, self.rews, self.dones, self.infos = {}, {}, {}, {}, {}

		""" Estimation related values """
		self.kernel = RBF(length_scale=env_config['kernel_length_scale'])
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel, optimizer=None)
		self.mu_model = None
		self.uncertainty_model = None

	def step(self, action_dict):
		""" Move the agents, take samples and process the new state. """

		assert self.resetted, "ERROR: The environment must be reseted first!"

		for agent_id, agent in zip(self._agent_ids, self.agents):
			# Execute every action and get the measurement #
			self.measurements[agent_id], _, self.dones[agent_id], self.infos[agent_id] = agent.step(action_dict[agent_id])

		# Update the model and return the map and the uncertainty #
		self.mu_model, self.uncertainty_model = self.update_model(self.measurements)





