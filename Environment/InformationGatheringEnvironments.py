import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class QuadcopterAgent:
	_action_types = ['CartesianDiscrete', 'AngleAndDistance']

	agent_config_template = {'navigation_map': None,
	                         'mask_size': None,
	                         'initial_position': None,
	                         'speed': None,
	                         'max_illegal_movements': None,
	                         'max_distance': None,
	                         'ground_truth_field': None,
	                         'dt': None}

	def __init__(self, agent_config: dict, agent_id, initial_position = None):

		""" Quadcopter Agent Class:
		 Serves as an abstract class for the vehicle movement processing.
		 It defines the individual mesurement process.  """

		self.done = None
		self.agent_id = agent_id
		self.navigation_map = agent_config['navigation_map']

		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))

		if initial_position is not None:
			self.initial_position = initial_position
		else:
			self.initial_position = agent_config['initial_position']

		# A dictionary with the action parameters #
		# Initial position #
		self.position = None
		# Waypoints #
		self.next_wp = None
		self.director_vector = None
		self.wp_reached = False
		# Initial measurement mask #
		self.measurement = None
		self.mask_size = agent_config['mask_size']
		self.speed = agent_config['speed']

		# Other variables #
		self.illegal_movements = 0
		self.max_illegal_movements = agent_config['max_illegal_movements']
		self.max_distance = agent_config['max_distance']
		self.distance = 0
		self._ground_truth_field = agent_config['ground_truth_field']
		self.dt = agent_config['dt']

		self.fig = None

	@property
	def ground_truth_field(self):
		return self._ground_truth_field

	@ground_truth_field.setter
	def ground_truth_field(self, new_value):
		self._ground_truth_field = new_value

	def go_to_next_waypoint_relative(self, angle, distance):
		""" Set the next waypoint relative to the current position """

		next_attempted_wp = self.position + distance * np.asarray([np.cos(angle), np.sin(angle)])
		# Check if the next wp is navigable
		valid = self.check_movement_feasibility(next_attempted_wp)

		if not valid:
			return 0

			self.wp_reached = True
			self.next_wp = np.copy(self.position)

		self.next_wp = next_attempted_wp
		self.director_vector = (self.next_wp - self.position) / np.linalg.norm(self.next_wp - self.position)
		self.wp_reached = False

		return 1

	def stop_go_to(self):
		""" Interrupt the current waypoint """
		next_attempted_wp = self.position
		self.wp_reached = True

	def step(self):
		""" Take 1 step  """

		if not self.wp_reached:
			# Compute the new position #
			d_pos = self.speed * self.director_vector * self.dt
			self.distance += np.linalg.norm(d_pos)

			if self.check_movement_feasibility(self.position + d_pos):
				# Check if there is any collisions #
				self.position = self.position + d_pos
			else:
				# If indeed there is a collision, stop and update the measurement #
				self.illegal_movements += 1
				self.wp_reached = True
				self.measurement = self.take_measurement()
				self.next_wp = np.copy(self.position)  # Set the next_wp in the pre-collision position #

			if np.linalg.norm(self.position - self.next_wp) < np.linalg.norm(d_pos) and not self.wp_reached:
				self.wp_reached = True
				self.next_wp = np.copy(self.position)
				self.measurement = self.take_measurement()

		self.done = self.illegal_movements >= self.max_illegal_movements or self.distance > self.max_distance

		return {'data': self.measurement, 'position': self.position}, self.done

	def check_movement_feasibility(self, pos):
		""" Check if the movement is possible with the current navigation map.
			This method should be overriden when using the real deploy agent. """

		in_bounds = all(pos <= self.navigation_map.shape) and all(pos > 0)

		if in_bounds:
			return self.navigation_map[int(pos[0]), int(pos[1])] == 1
		else:
			return False

	def take_measurement(self, pos=None):

		if pos is None:
			pos = self.position

		pos = pos.astype(int)

		upp_lims = np.clip(pos + self.mask_size, 0, self.navigation_map.shape)
		low_lims = np.clip(pos - self.mask_size, 0, self.navigation_map.shape)

		measurement_mask = self._ground_truth_field[low_lims[0]: upp_lims[0] + 1, upp_lims[1]: upp_lims[1] + 1]

		return measurement_mask

	def reset(self):
		""" Reset the positions, measuremente, and so on. """
		
		self.done = False

		if self.initial_position is None:
			self.position = self.visitable_positions[np.random.randint(0, len(self.visitable_positions))]
		else:
			self.position = np.copy(self.initial_position)

		self.measurement = self.take_measurement()
		self.distance = 0
		self.illegal_movements = 0
		self.wp_reached = True
		self.next_wp = np.copy(self.position)

		return {'data': self.measurement, 'position': self.position}

	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 2)

			self.axs[0].imshow(self.navigation_map, cmap='gray')
			self.d_pos, = self.axs[0].plot(self.position[1], self.position[0], 'rx')
			self.d_wp, = self.axs[0].plot(self.next_wp[1], self.next_wp[0], 'bo')
			self.d_sense = self.axs[1].imshow(self.measurement)
			self.d_mask = self.axs[0].add_patch(
				Rectangle((self.position[1] - self.mask_size[1] / 2, self.position[0] - self.mask_size[0] / 2),
				          self.mask_size[1], self.mask_size[0], fc='g', ec='g', lw=2, ))

		else:

			self.d_pos.set_xdata(self.position[1])
			self.d_pos.set_ydata(self.position[0])
			self.d_wp.set_xdata(self.next_wp[1])
			self.d_wp.set_ydata(self.next_wp[0])
			self.d_sense.set_data(self.measurement)
			self.d_mask.set_xy((self.position[1] - self.mask_size[1] / 2, self.position[0] - self.mask_size[0] / 2))

			self.fig.canvas.draw()


class SynchronousMultiAgentIGEnvironment(MultiAgentEnv):

	def __init__(self, env_config):

		super().__init__()

		self.env_config = env_config

		# Create the agents #
		self.agentID = 0
		self._agent_ids = []
		self.agents = {}
		for _ in range(env_config['number_of_agents']):
			self.spawn_agent()

		self.dones = [False] * env_config['number_of_agents']
		# List of ready agents #
		self.agents_ready = [True] * env_config['number_of_agents']

		# Create space-state sets
		self.observation_space = gym.spaces.Box(low=0.0, high=10000, shape=(5, *env_config['navigation_map'].shape))
		self.action_space = gym.spaces.Discrete(env_config['number_of_actions'])

		# Other variables
		self.visitable_positions = np.column_stack(np.where(env_config['navigation_map'] == 1))
		self.resetted = False
		self.measurements = [None] * env_config['number_of_actions']
		self.valids = [None] * env_config['number_of_actions']
		self.rewards = {}
		self.states = {}

		""" Regression related values """
		self.kernel = RBF(length_scale=env_config['kernel_length_scale'], length_scale_bounds=(0.1, 10))
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=20, alpha=0.05)
		self.measured_values = None
		self.measured_locations = None
		self.mu = None
		self.uncertainty = None
		self.Sigma = None
		self.H_ = 0
		self.fig = None

	def action2angle(self, action):

		return 2 * np.pi * action / self.env_config['number_of_actions']

	def spawn_agent(self):

		agentID = self.agentID
		self.agents[agentID] = QuadcopterAgent(self.env_config['agent_config'], agentID, self.env_config['initial_positions'][agentID])
		self._agent_ids.append(agentID)
		self.agentID += 1

	def step(self, action_dict):
		""" Process actions for ALL agents """

		assert self.resetted, "You need to call env.reset() first!"

		for i, agent in self.agents.items():
			action = action_dict[i]

			# Transfor discrete action into an angle
			ang = self.action2angle(action)

			# Schedule every waypoint
			agent.go_to_next_waypoint_relative(angle=ang, distance=self.env_config['meas_distance'])

		# Move environment until all target positions are reached
		wp_reached_vec = [False] * self.env_config['number_of_agents']

		while not all(wp_reached_vec):

			# One step per agent
			for i, agent in self.agents.items():

				meas, ended = agent.step()

				# Check if reached
				if agent.wp_reached and not wp_reached_vec[i]:
					wp_reached_vec[i] = True
					self.measurements[i] = meas  # Copy measurement if so

					# Check if end of mission
					if ended:
						self.dones[i] = True

			self.render()

			plt.pause(0.001)

		# Once every agent has reached its destination, compute the reward and the state #

		self.mu, self.uncertainty, self.Sigma = self.update_model()

		self.rewards = self.reward_function()

		self.states = self.update_states()

		self.dones = {i: self.agents[i].done for i in self._agent_ids}


		return self.states, self.rewards, self.dones, {}

	def update_model(self):

		""" Fit the gaussian process using the measurements and
		 return a new inferred map and its uncertainty """

		# Append the new data #
		if self.measured_locations is None:
			self.measured_locations = np.asarray([meas['position'] for meas in self.measurements.values()])
			self.measured_values = np.asarray([np.nanmean(meas['data']) for meas in self.measurements.values()])
		else:
			self.measured_locations = np.vstack((self.measured_locations,
			                                     np.asarray([meas['position'] for meas in self.measurements.values()])))
			self.measured_values = np.hstack((self.measured_values,
			                                  np.asarray([np.nanmean(meas['data']) for meas in self.measurements.values()])))

		unique_locs, unique_indxs = np.unique(self.measured_locations, axis=0, return_index=True)
		self.GaussianProcess.fit(unique_locs, self.measured_values[unique_indxs])

		mu, Sigma = self.GaussianProcess.predict(self.visitable_positions, return_cov=True)
		uncertainty = Sigma.diagonal()

		mu_map = np.copy(self.env_config['navigation_map'])
		mu_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = mu

		uncertainty_map = np.copy(self.env_config['navigation_map'])
		uncertainty_map[self.visitable_positions[:,0], self.visitable_positions[:,1]] = uncertainty

		return mu_map, uncertainty_map, Sigma

	def reward_function(self):

		Tr = self.Sigma.trace()

		rew = self.H_ - Tr

		self.H_ = Tr

		return {i: rew for i in self._agent_ids}

	def update_states(self):
		""" Update the states """

		states = {i: self.individual_state(i) for i in self._agent_ids}

		return states

	def individual_state(self, agent_indx):

		# First channel: navigation/obstacle map
		nav_map = np.copy(self.env_config['navigation_map'])

		# Second channel: self-position map
		position_map = np.zeros_like(self.env_config['navigation_map'])
		agent_position = self.agents[agent_indx].position.astype(int)
		position_map[agent_position[0], agent_position[1]] = 1.0

		# Third channel: Other agents channel
		other_agents_map = np.zeros_like(self.env_config['navigation_map'])
		other_agents_ids = np.copy(self._agent_ids)
		other_agents_ids = np.delete(other_agents_ids,agent_indx)

		for agent_id in other_agents_ids:
			agent_position = self.agents[agent_id].position.astype(int)
			other_agents_map[agent_position[0], agent_position[1]] = 1.0

		return np.concatenate((nav_map[np.newaxis],
		                       position_map[np.newaxis],
		                       other_agents_map[np.newaxis],
		                       self.mu[np.newaxis],
		                       self.uncertainty[np.newaxis]))

	def reset(self):

		self.resetted = True

		# Reset the vehicles and take the first measurements #
		self.measurements = {i: agent.reset() for i, agent in self.agents.items()}

		# Update the model with the initial values #
		self.mu, self.uncertainty, self.Sigma = self.update_model()

		# Update the states
		self.states = self.update_states()

		return self.states

	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 2)

			self.axs[0].imshow(self.env_config['navigation_map'], cmap='gray')
			self.d_pos, = self.axs[0].plot([agent.position[1] for agent in self.agents.values()], [agent.position[0] for agent in self.agents.values()], 'rx')
			self.d_mu = self.axs[1].imshow(self.mu, cmap='jet')

		else:

			self.d_pos.set_xdata([agent.position[1] for agent in self.agents.values()])
			self.d_pos.set_ydata([agent.position[0] for agent in self.agents.values()])
			self.d_mu.set_data(self.mu)

			self.fig.canvas.draw()


class AsyncronousMultiAgentIGEnvironment(SynchronousMultiAgentIGEnvironment, MultiAgentEnv):

	def __init__(self, env_config):

		super(AsyncronousMultiAgentIGEnvironment, self).__init__(env_config)

		self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

	def action2angdistance(self, action):

		action = (action + 1) / 2  # From [-1,1] to [0,1]
		action[0] = action[0] * 2 * np.pi
		action[1] = action[1] * (self.env_config['max_meas_distance'] - self.env_config['min_meas_distance']) + \
		            self.env_config['min_meas_distance']

		return action

	def step(self, action_dict):
		""" Process actions ONLY for AVAILABLE agents """

		assert self.resetted, "You need to call env.reset() first!"

		# Reset the internal states and meas
		self.states = {}
		self.rewards = {}
		self.measurements = {}

		# First, query the actions #
		for i, action in action_dict.items():

			# Transfor discrete action into an angle
			ang, dist = self.action2angdistance(action)

			# Schedule every waypoint
			self.agents[i].go_to_next_waypoint_relative(angle=ang, distance=dist)

		# Move environment until at least one agent have finished its action
		wp_reached_vec = [False] * self.env_config['number_of_agents']

		while not any(wp_reached_vec):

			# One step per agent
			for i, agent in self.agents.items():

				meas, ended = agent.step()

				# Check if reached
				if agent.wp_reached and not wp_reached_vec[i]:
					wp_reached_vec[i] = True
					self.measurements[i] = meas  # Copy measurement if so
					self.states[i] = self.individual_state(i)
					self.mu, self.uncertainty, self.Sigma = self.update_model()
					self.rewards[i] = self.reward_function()

			env.render()
			plt.pause(0.1)

		self.dones = {i: agent.done for i, agent in self.agents.items()}


		return self.states, self.rewards, self.dones, {}


if __name__ == '__main__':
	from deap import benchmarks



	""" Benchmark parameters """
	A = [[0.5, 0.5],
	     [0.25, 0.25],
	     [0.25, 0.75],
	     [0.75, 0.25],
	     [0.75, 0.75]]

	C = [0.005, 0.002, 0.002, 0.002, 0.002]


	def shekel_arg0(sol):
		return benchmarks.shekel(sol, A, C)[0]


	def r_interest(val_mu):

		return np.max((0, 1 + val_mu))


	""" Compute Ground Truth """
	navigation_map = np.genfromtxt('wesslinger_map.txt')
	X = np.linspace(0, 1, navigation_map.shape[1])
	Y = np.linspace(0, 1, navigation_map.shape[0])
	X, Y = np.meshgrid(X, Y)
	Z = np.fromiter(map(shekel_arg0, zip(X.flat, Y.flat)), dtype=float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
	Z = (Z - Z.min()) / (Z.max() - Z.min() + 1E-8)



	agent_config = {'navigation_map': navigation_map,
	                'mask_size': (0, 0),
	                'initial_position': None,
	                'speed': 2,
	                'max_illegal_movements': 10,
	                'max_distance': 10000000,
	                'ground_truth_field': Z,
	                'dt': 1}


	my_env_config = {'number_of_agents': 3,
	                 'number_of_actions': 8,
	                 'kernel_length_scale': 0.2,
	                 'agent_config': agent_config,
	                 'navigation_map': navigation_map,
	                 'meas_distance': 1,
	                 'initial_positions': np.array([[21,14],[30,16],[36,41]]),
	                 'max_meas_distance': 5,
	                 'min_meas_distance': 1
	                 }

	#env = SynchronousMultiAgentIGEnvironment(env_config=my_env_config)
	env = AsyncronousMultiAgentIGEnvironment(env_config=my_env_config)


	s = env.reset()

	dones = [False] * 4


	while not all(dones):

		action = {i: env.action_space.sample() for i in env._agent_ids if i in s.keys()}


		agents_ready_for_action = [i for i in s.keys()]
		print("The agents", agents_ready_for_action, "are ready for action!")

		s, r, dones, _ = env.step(action)

		dones = list(dones.values())






