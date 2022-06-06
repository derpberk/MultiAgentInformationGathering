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

	def __init__(self, agent_config: dict, agent_id):

		""" Quadcopter Agent Class:
		 Serves as an abstract class for the vehicle movement processing.
		 It defines the individual mesurement process.  """

		self.agent_id = agent_id
		self.navigation_map = agent_config['navigation_map']

		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))
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
		self.director_vector = (self.next_wp - self.position)/np.linalg.norm(self.next_wp - self.position)
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
				self.position =  self.position + d_pos
			else:
				# If indeed there is a collision, stop and update the measurement #
				self.illegal_movements += 1
				self.wp_reached = True
				self.measurement = self.take_measurement()
				self.next_wp = np.copy(self.position) # Set the next_wp in the pre-collision position #

			if np.linalg.norm(self.position - self.next_wp) < np.linalg.norm(d_pos) and not self.wp_reached:

				self.wp_reached = True
				self.next_wp = np.copy(self.position)
				self.measurement = self.take_measurement()

		done = self.illegal_movements >= self.max_illegal_movements or self.distance > self.max_distance

		return {'data': self.measurement, 'position': self.position}, done

	def check_movement_feasibility(self, pos):
		""" Check if the movement is possible with the current navigation map.
			This method should be overriden when using the real deploy agent. """
		visitable = self.navigation_map[int(pos[0]), int(pos[1])] == 1
		in_bounds = all(pos <=self.navigation_map.shape) and all(pos > 0)

		return visitable and in_bounds

	def take_measurement(self, pos = None):

		if pos is None:
			pos = self.position

		pos = pos.astype(int)

		measurement_mask = self._ground_truth_field[pos[0] - self.mask_size[0]: pos[0] + self.mask_size[0] + 1,
		                                            pos[1] - self.mask_size[1]: pos[1] + self.mask_size[1] + 1]

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
		self.wp_reached = True
		self.next_wp = np.copy(self.position)

		return self.measurement

	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(1,2)

			self.axs[0].imshow(self.navigation_map, cmap = 'gray')
			self.d_pos, = self.axs[0].plot(self.position[1], self.position[0], 'rx')
			self.d_wp, = self.axs[0].plot(self.next_wp[1], self.next_wp[0], 'bo')
			self.d_sense = self.axs[1].imshow(self.measurement)
			self.d_mask = self.axs[0].add_patch(Rectangle((self.position[1]-self.mask_size[1]/2, self.position[0]-self.mask_size[0]/2), self.mask_size[1], self.mask_size[0], fc ='g',ec ='g', lw = 2, ))

		else:

			self.d_pos.set_xdata(self.position[1])
			self.d_pos.set_ydata(self.position[0])
			self.d_wp.set_xdata(self.next_wp[1])
			self.d_wp.set_ydata(self.next_wp[0])
			self.d_sense.set_data(self.measurement)
			self.d_mask.set_xy((self.position[1]-self.mask_size[1]/2, self.position[0]-self.mask_size[0]/2))

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

		self.dones = [False]*env_config['number_of_agents']
		# List of ready agents #
		self.agents_ready = [True]*env_config['number_of_agents']

		# Create space-state sets
		self.observation_space = gym.spaces.Box(low=0.0, high=10000, shape=(5, *env_config['navigation_map'].shape))
		self.action_space = gym.spaces.Discrete(env_config['number_of_actions'])

		# Other variables
		self.resetted = False
		self.measurements = [None] * env_config['number_of_actions']
		self.valids = [None] * env_config['number_of_actions']
		self.rewards = {}
		self.states = {}

		""" Regression related values """
		self.kernel = RBF(length_scale=env_config['kernel_length_scale'])
		self.GaussianProcess = GaussianProcessRegressor(kernel=self.kernel, optimizer=None)
		self.measured_values = None
		self.measured_locations = None
		self.mu = None
		self.uncertainty = None
		self.uncertainty = None
		self.Sigma = None
		self.H_ = 0

	def action2angle(self, action):

		return 2*np.pi * action/self.env_config['number_of_actions']

	def spawn_agent(self):

		agentID = self.agentID
		self.agents[agentID] = QuadcopterAgent(self.env_config['agent_config'], agentID)
		self._agent_ids.append(agentID)
		self.agentID += 1

	def step(self, action_dict):
		""" Process actions for agents that are available """

		assert self.resetted, "You need to call env.reset() first!"

		for i, agent in self.agents:

			action = action_dict[self.agents_ids_dict[i]]

			# Transfor discrete action into an angle
			ang = self.action2angle(action)

			# Schedule every waypoint
			agent.go_to_next_waypoint_relative(angle=ang, distance=self.env_config['meas_distance'])


		# Move environment until all target positions are reached
		wp_reached_vec = [False] * self.env_config['number_of_actions']
		while not all(wp_reached_vec):

			# One step per agent
			for i in self._agent_ids:
				meas, ended = self.agent[i].step()

				# Check if reached
				if self.agents[i].wp_reached:
					wp_reached_vec[i] = True
					self.measurements[i] = np.copy(meas) # Copy measurement if so
				# Check if end of mission
				if ended:
					self.dones[i] = True

		# Once every agent has reached its destination, compute the reward and the state #

		self.mu, self.uncertainty, self.Sigma = self.update_model(self.measurements)

		self.rewards = self.reward_function()

		self.states = self.update_states()

		self.dones = {i: self.agents[i].done for i in self._agent_ids}


		return self.states, self.rewards, self.dones, {}

	def update_model(self):

		""" Fit the gaussian process using the measurements and
		 return a new inferred map and its uncertainty """

		# Append the new data #
		if self.measured_locations is None:
			self.measured_locations = np.asarray([self.measurements[ind]['position'] for ind in self._agent_ids])
			self.measured_values = np.asarray([np.mean(self.measurements[ind]['data']) for ind in self._agent_ids])
		else:
			self.measured_locations = np.vstack((self.measured_locations, np.asarray([self.measurements[ind]['position'] for ind in self._agent_ids])))
			self.measured_values = np.vstack((self.measured_values, np.asarray([np.mean(self.measurements[ind]['data']) for ind in self._agent_ids])))

		self.GaussianProcess.fit(self.measured_locations, self.measured_values)

		mu, Sigma = self.GaussianProcess.predict(self.visitable_positions, return_cov=True)
		uncertainty = Sigma.diagonal()

		mu_map = np.copy(self.env_config['navigation_map'])
		mu_map[self.visitable_position] = mu

		uncertainty_map = np.copy(self.env_config['navigation_map'])
		uncertainty_map[self.visitable_position] = uncertainty

		return mu, uncertainty, Sigma

	def reward_function(self):

		Tr = self.Sigma.trace()

		rew =  self.H_ - Tr

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
		other_agents_ids.pop(agent_indx)

		for agent_id in other_agents_ids:

			agent_position = self.agents[agent_id].position.astype(int)
			other_agents_map[agent_position[0], agent_position[1]] = 1.0


		return np.concatenate((nav_map[np.newaxis],
		                       agent_position[np.newaxis],
		                       other_agents_map[np.newaxis],
		                       self.mu[np.newaxis],
		                       self.uncertainty[np.newaxis]))

	def reset(self):

		self.resetted = True

		# Reset the vehicles and take the first measurements #
		self.measurements = [agent.reset() for agent in self.agents]

		# Update the model with the initial values #
		self.mu, self.uncertainty, self.Sigma = self.update_model(new_measurements=self.measurements)

		# Update the states
		self.states = self.update_states()

		return self.states

class AsyncronousMultiAgentIGEnvironment(SincronousMultiAgentIGEnvironment, MultiAgentEnv):

	def __init__(self):

		super(AsyncronousMultiAgentIGEnvironment, self).__init__()

	def action2angdistance(self, action):

		action = (action + 1)/2 # From [-1,1] to [0,1]
		action[1] = action[1]*2*np.pi
		action[2] = action[1] * (self.env_config['max_meas_distance'] - self.env_config['min_meas_distance']) + self.env_config['min_meas_distance']

		return action


	def step(self, action_dict):
		""" Process actions for agents that are available """

		assert self.resetted, "You need to call env.reset() first!"



		for i, action in action_dict.keys():

			# Transfor discrete action into an angle
			ang = self.action2angdistance(action)

			# Schedule every waypoint
			self.agent[i].go_to_next_waypoint_relative(angle=ang, distance=self.env_config['meas_distance'])

		wp_reached_vec = [agent.wp_reached for agent in self.agents]
		while not any(wp_reached_vec):

			# One step per agent
			for i in self._agent_ids:
				meas, ended = self.agent[i].step()

				# Check if reached
				if self.agents[i].wp_reached:
					wp_reached_vec[i] = True
					self.measurements[i] = np.copy(meas)  # Copy measurement if so
				# Check if end of mission
				if ended:
					self.dones[i] = True

		# Once every agent has reached its destination, compute the reward and the state #

		self.mu, self.uncertainty, self.Sigma = self.update_model()

		self.rewards = self.reward_function()

		self.states = self.update_states()

		self.dones = {i: self.agents[i].done for i in self._agent_ids}

		return self.states, self.rewards, self.dones, {}



if __name__ == '__main__':


	""" Test for one agent """

	agent_config = {'navigation_map': np.ones((100,100)),
	                'mask_size': (10,10),
	                'initial_position': np.array([50,50]),
	                'speed': 1,
	                'max_illegal_movements': 10,
	                'max_distance': 10000000,
	                'ground_truth_field': np.random.rand(100,100),
	                'dt': 1}

	agent = QuadcopterAgent(agent_config, 'pepe')

	agent.reset()
	agent.render()

	def target_to_angledist(current, target):

		diff = target - current
		angle = np.arctan2(diff[1], diff[0])
		dist = np.linalg.norm(diff)

		return angle,dist

	def onclick(event):

		global agent

		agent.stop_go_to()
		new_position = np.asarray([event.ydata, event.xdata])
		angle,dist = target_to_angledist(agent.position, new_position)
		agent.go_to_next_waypoint_relative(angle, dist)


	cid = agent.fig.canvas.mpl_connect('button_press_event', onclick)

	done = False

	indx = 0

	x,y = np.meshgrid(np.arange(10,80,10), np.arange(10,80,10))

	wps = np.column_stack((y.flatten(),x.flatten()))

	indx = 0

	print("Waypoint ", wps[indx,:])
	angle, dist = target_to_angledist(agent.position, wps[indx,:])
	agent.go_to_next_waypoint_relative(angle, dist)

	while True:


		_,done = agent.step()
		agent.render()
		plt.pause(0.01)


		if indx == len(wps) - 1:
			agent.reset()
			indx = 0

		if agent.wp_reached:

			indx += 1
			print("Waypoint ", wps[indx, :])
			angle, dist = target_to_angledist(agent.position, wps[indx,:])
			agent.go_to_next_waypoint_relative(angle, dist)
