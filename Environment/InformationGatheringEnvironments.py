import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from OilSpillEnvironment import OilSpillEnv
from ShekelGroundTruth import Shekel


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
		self._ground_truth_field = None
		self.dt = agent_config['dt']
		self.collision = False
		self.fig = None

	@property
	def ground_truth_field(self):
		return self._ground_truth_field

	@ground_truth_field.setter
	def ground_truth_field(self, new_value):
		self._ground_truth_field = new_value

	def go_to_next_waypoint_relative(self, angle, distance):
		""" Set the next waypoint relative to the current position.
		 Return True if feasible, False otherwise. """

		# Compute the next attemted position
		next_attempted_wp = self.position + distance * np.asarray([np.cos(angle), np.sin(angle)])
		# Check if the next wp is navigable
		valid = self.check_movement_feasibility(next_attempted_wp)

		# If the WP is not valid, end the action
		if not valid:

			self.wp_reached = True
			self.next_wp = np.copy(self.position)
			self.collision = True # Set the collision state
			self.illegal_movements += 1

			return True

		# If the WP is acceptable, set the target
		self.next_wp = next_attempted_wp
		self.director_vector = (self.next_wp - self.position) / np.linalg.norm(self.next_wp - self.position)
		self.wp_reached = False
		self.collision = False

		return False

	def stop_go_to(self):
		""" Interrupt the current waypoint """
		self.next_wp = self.position
		self.wp_reached = True

	def step(self):
		""" Integrate the speed to obtain the next position  """

		# If the WP is not reached, and not in a collision state, compute the dynamic
		if not self.wp_reached and not self.collision:

			# Compute the new position #
			d_pos = self.speed * self.director_vector * self.dt
			self.distance += np.linalg.norm(d_pos)

			# Check if the new position is feasible
			if self.check_movement_feasibility(self.position + d_pos):
				self.position = self.position + d_pos
				self.collision = False
			else:
				# If indeed there is a collision, stop and update the measurement #
				self.illegal_movements += 1
				self.wp_reached = True
				self.measurement = self.take_measurement()
				self.next_wp = np.copy(self.position)  # Set the next_wp in the pre-collision position #
				self.collision = True

			if np.linalg.norm(self.position - self.next_wp) < np.linalg.norm(d_pos) and not self.wp_reached:
				self.wp_reached = True
				self.next_wp = np.copy(self.position)
				self.measurement = self.take_measurement()

		self.done = self.illegal_movements >= self.max_illegal_movements or self.distance > self.max_distance

		# Return the data, the position, the end-mission condition and the collision state #
		return {'data': self.measurement, 'position': self.position}, self.done, self.collision

	def check_movement_feasibility(self, pos):
		""" Check if the movement is possible with the current navigation map.
			This method should be overriden when using the real deploy agent. """

		in_bounds = all(pos <= self.navigation_map.shape) and all(pos > 0)

		if in_bounds:

			return self.navigation_map[(pos[0]).astype(int), (pos[1]).astype(int)] == 1
		else:
			return False

	def take_measurement(self, pos=None):

		assert self._ground_truth_field is not None, 'The ground truth is not set!'

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
		self.collision = False

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

		self.mse = None
		self.max_sensed_value = None
		self.regret = None
		self._eval = False
		self.env_config = env_config

		# Create the agents #
		self.agentID = 0
		self._agent_ids = []
		self.agents = {}
		for _ in range(env_config['number_of_agents']):
			self.spawn_agent()

		self.dones = None
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
		self.infos = {}

		""" Regression related values """
		if env_config['dynamic'] == 'OilSpillEnv':
			self.ground_truth = OilSpillEnv(self.env_config['navigation_map'], dt=1, flow=10, gamma=1, kc=1, kw=1)
		elif env_config['dynamic'] == 'Shekel':
			self.ground_truth = Shekel(1 - self.env_config['navigation_map'], 1, max_number_of_peaks=4, is_bounded=True, seed=0)
		else:
			raise NotImplementedError("This benchmark is not implemented")

		self.kernel = RBF(length_scale=env_config['kernel_length_scale'], length_scale_bounds='fixed')
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel, alpha=0.01, optimizer=None)
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

	@property
	def eval(self):
		return self._eval

	@eval.setter
	def eval(self, flag):
		self._eval = flag

	def step(self, action_dict):
		""" Process actions for ALL agents """

		assert self.resetted, "You need to call env.reset() first!"

		# Set mission for every
		for i, agent in self.agents.items():

			action = action_dict[i]

			# Transfor discrete action into an angle
			ang = self.action2angle(action)

			# Schedule every waypoint
			agent.go_to_next_waypoint_relative(angle=ang, distance=self.env_config['meas_distance'])

		# Time to move the agents!
		# Move environment until all target positions are reached
		wp_reached_vec = [agent.wp_reached for agent in self.agents.values()]

		while not all(wp_reached_vec):

			# One step per agent
			for i, agent in self.agents.items():

				# Process the movement of the agent
				# Note that internally, nothing is done if the agent is in collision state #
				meas, ended, collision = agent.step()

				# Check if reached
				if agent.wp_reached and not wp_reached_vec[i]:

					# If the agent reached the waypoint...
					wp_reached_vec[i] = True
					self.measurements[i] = meas  # Copy measurement if so

					# Check end-of-mission
					self.dones[i] = ended

		# Once every agent has reached its destination, compute the reward and the state #

		self.mu, self.uncertainty, self.Sigma = self.update_model()

		self.rewards = self.reward_function()

		self.states = self.update_states()

		self.dones = {i: self.agents[i].done for i in self._agent_ids}



		# Update the ground truth state and pass the field to agents #
		self.ground_truth.step()
		self.update_vehicles_ground_truths()


		return self.states, self.rewards, self.dones, {'collisions': [agent.collision for agent in self.agents.values()]}

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


		# Obtain the max value obtained by the fleet to compute the regret #
		self.max_sensed_value = self.measured_values.max()
		self.regret = 1 - self.measured_values[-self.env_config['number_of_agents']:] - self.max_sensed_value

		"""
		unique_locs, unique_indxs = np.unique(self.measured_locations, axis=0, return_index=True)
		unique_values = self.measured_values[unique_indxs]
		"""

		self.GaussianProcess.fit(self.measured_locations, self.measured_values)

		mu, Sigma = self.GaussianProcess.predict(self.visitable_positions, return_cov=True)

		if self._eval:
			self.mse = mean_squared_error(y_true = self.ground_truth.ground_truth_field[self.visitable_positions[:,0], self.visitable_positions[:,1]],
			                              y_pred = mu,
			                              squared=True)

		uncertainty = np.sqrt(Sigma.diagonal())

		mu_map = np.copy(self.env_config['navigation_map'])
		mu_map[self.visitable_positions[:, 0],
		       self.visitable_positions[:, 1]] = mu

		uncertainty_map = np.copy(self.env_config['navigation_map'])
		uncertainty_map[self.visitable_positions[:, 0],
		                self.visitable_positions[:, 1]] = uncertainty

		return mu_map, uncertainty_map, Sigma

	def reward_function(self):
		""" The reward function has the following terms:
			H -> the decrement of the uncertainty from the previous instant to the next one. Defines the exploratory reward.
			"""
		Tr = np.trace(self.Sigma)/8

		uncertainty_component = (self.H_ - Tr)/self.env_config['number_of_agents']
		regret_component = 1 - np.clip(self.regret, 0.0, 1.0)

		reward = uncertainty_component*regret_component

		self.H_ = Tr

		return {i: reward[i] for i in self._agent_ids}

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

	def get_agents_positions(self):

		return np.asarray([agent.position for agent in self.agents.values()])

	def get_valid_action_mask(self, agent_id = 0):

		angles = np.asarray(list(map(self.action2angle, np.arange(0,self.env_config['number_of_actions']))))
		next_positions = self.agents[agent_id].position + 1.41*self.env_config['meas_distance'] * np.asarray([np.cos(angles), np.sin(angles)]).T

		return [self.agents[agent_id].check_movement_feasibility(pos) for pos in next_positions]


	def update_vehicles_ground_truths(self):

		for vehicle in self.agents.values():
			vehicle.ground_truth_field = self.ground_truth.ground_truth_field

	def reset(self):

		self.resetted = True

		self.dones = {i: False for i in self.agents.keys()}

		# Reset the ground truth and set the value for
		self.ground_truth.reset()
		self.update_vehicles_ground_truths()

		# Reset the model #
		self.measured_locations = None
		self.measured_values = None
		self.mu = None
		self.uncertainty = None
		self.Sigma = None

		# Reset the vehicles and take the first measurements #
		self.measurements = {i: agent.reset() for i, agent in self.agents.items()}

		# Update the model with the initial values #
		self.mu, self.uncertainty, self.Sigma = self.update_model()

		self.H_ = np.trace(self.Sigma)/8

		# Update the states
		self.states = self.update_states()


		return self.states

	def render(self, mode='human'):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 4)

			self.axs[0].imshow(self.env_config['navigation_map'], cmap='gray')
			self.d_pos, = self.axs[0].plot([agent.position[1] for agent in self.agents.values()], [agent.position[0] for agent in self.agents.values()], 'rx')
			self.d_mu = self.axs[1].imshow(self.mu, cmap='jet', vmin=0, vmax=self.ground_truth.ground_truth_field.max(), interpolation='bicubic')
			self.d_unc = self.axs[2].imshow(self.uncertainty, cmap = 'gray_r')
			self.d_gt = self.axs[3].imshow(self.ground_truth.ground_truth_field, cmap='jet', vmin=0, vmax=self.ground_truth.ground_truth_field.max() , interpolation='bicubic')

		else:

			self.d_pos.set_xdata([agent.position[1] for agent in self.agents.values()])
			self.d_pos.set_ydata([agent.position[0] for agent in self.agents.values()])
			self.d_mu.set_data(self.mu)
			self.d_unc.set_data(self.uncertainty)
			self.d_gt.set_data(self.ground_truth.ground_truth_field)

			self.fig.canvas.draw()

	def check_actions(self, act_dict):
		""" Check if the action constitutes a collision for every agent """

		is_valid = []

		for agent_id, a in act_dict.items():
			# Compute the action angle #
			angle = self.action2angle(a)
			# Compute the new position #
			next_attempted_wp = self.agents[agent_id].position + self.env_config['meas_distance'] * np.asarray([np.cos(angle), np.sin(angle)])
			# Check if the new position is feasible
			is_valid.append(self.agents[agent_id].check_movement_feasibility(next_attempted_wp))

		return is_valid



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
		wp_reached_vec = [agent.wp_reached for agent in self.agents.values()]

		while not any(wp_reached_vec):

			# One step per agent
			for i, agent in self.agents.items():

				meas, ended, collision = agent.step()

				# Check if reached
				if agent.wp_reached and not wp_reached_vec[i]:
					wp_reached_vec[i] = True
					self.measurements[i] = meas  # Copy measurement if so
					self.states[i] = self.individual_state(i)
					self.mu, self.uncertainty, self.Sigma = self.update_model()
					self.rewards[i] = self.reward_function()

		self.dones = {i: agent.done for i, agent in self.agents.items()}


		return self.states, self.rewards, self.dones, {'collisions': [agent.collision for agent in self.agents.values()]}


if __name__ == '__main__':
	from deap import benchmarks

	np.random.seed(11)


	""" Compute Ground Truth """
	navigation_map = np.genfromtxt('wesslinger_map.txt')

	agent_config = {'navigation_map': navigation_map,
	                'mask_size': (1, 1),
	                'initial_position': None,
	                'speed': 2,
	                'max_illegal_movements': 10,
	                'max_distance': 200,
	                'dt': 1}


	my_env_config = {'number_of_agents': 3,
	                 'number_of_actions': 8,
	                 'kernel_length_scale': 2,
	                 'agent_config': agent_config,
	                 'navigation_map': navigation_map,
	                 'meas_distance': 2,
	                 'initial_positions': np.array([[21,14],[30,16],[36,41]]),
	                 'max_meas_distance': 5,
	                 'min_meas_distance': 1,
	                 'dynamic': 'Shekel',
	                 }

	env = SynchronousMultiAgentIGEnvironment(env_config=my_env_config)
	#env = AsyncronousMultiAgentIGEnvironment(env_config=my_env_config)


	s = env.reset()

	_,ax = plt.subplots(1,1)

	dones = [False] * 4

	idx = 0
	action = {i: env.action_space.sample() for i in env._agent_ids if i in s.keys()}
	R = [0]
	t = [0]
	while not all(dones):


		idx+=1

		agents_ready_for_action = [i for i in s.keys()]
		print("The agents", agents_ready_for_action, "are ready for action!")

		s, r, dones, infos = env.step(action)

		R.append(r[0])
		t.append(t[-1]+1)

		print(f"Agent colliding: {infos['collisions']}")

		action = {i: env.action_space.sample() if infos['collisions'][i] else action[i] for i in env._agent_ids}

		dones = list(dones.values())

	plt.plot(t,R)
	plt.show()
	env.render()
	plt.show()





