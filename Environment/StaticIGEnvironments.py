import sys
sys.path.append('.')

import numpy as np
from Environment.SimpleFleet import Fleet
from gym.spaces import Discrete, Box
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as K, Matern, RBF, WhiteKernel as W
import matplotlib.pyplot as plt

class StaticIGEnv:
	""" Static Information Gathering Scenario """

	def __init__(self, 
				 number_of_vehicles: int,
				 navigation_map: np.ndarray,
				 max_travel_distance: float,
				 initial_vehicle_positions: np.ndarray,
				 movement_type: str,
				 movement_limits: tuple,
				 benchmark: object,
				 max_collisions: int,
				 # Model parameters #
				 kernel_type: str,
				 lengthscale: float,
				 # Reward style #
				 reward_type: str,
				 # Default parameters #
				 number_of_movements = 8,
				 ) -> None:
	
		self.number_of_agents = number_of_vehicles
		self.collisions = 0
		self.active_agents = [True for i in range(self.number_of_agents)]

		# Create the fleet #
		self.fleet = Fleet(initial_vehicle_positions=initial_vehicle_positions,
							max_travel_distance=max_travel_distance,
							navigation_map=navigation_map,
							number_of_agents=number_of_vehicles,
							)

		# Inherit the benchmark
		self.benchmark = benchmark

		# Movement related variables
		self.number_of_movements = number_of_movements
		self.movement_limits = movement_limits
		self.max_collisions = max_collisions
		self.navigation_map = navigation_map
		self.movement_type = movement_type

		# Reward type
		self.reward_type = reward_type
		
		# Create the variables related to the model #
		self.measured_values = None
		self.measured_locations = None
		self.intermediate_model_metrics = None

		# Create the model #
		kernel = Matern(length_scale=lengthscale) if kernel_type == 'Matern' else RBF(length_scale=lengthscale)
		self.kernel = K(1.0) * kernel + W(0.001)
		self.model = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, alpha=0.001)
		# Model variables 
		self.mu = np.zeros_like(self.navigation_map)
		self.sigma = np.zeros_like(self.navigation_map)
		self.visitable_positions = np.column_stack(np.where(self.navigation_map == 1))

		# Create the state and action #
		if self.movement_type == "DISCRETE":
			self.action_space = Discrete(self.number_of_movements)
		else:
			self.action_space = Box(low=-1.0, high=1.0, shape=(2,))

		self.state_space = Box(low=0.0, high=1.0, shape=(4 if reward_type == 'Uncertainty' else 5, *self.navigation_map.shape))
		self.fig = None
	
		
	def update_model(self, new_measurements: dict):
		""" Every measurement is a dictionary that has a new pack of sensored positions for every agent. """
		
		# Create a dictionary for intermediate values #
		intermediate_model_metrics = {}

		mu_array = self.mu[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]
		sigma_array = self.sigma[self.visitable_positions[:, 0], self.visitable_positions[:, 1]]

		# First, shuffle the measurements to avoid overfitting the order #
		agents_ids = list(new_measurements.keys())
		np.random.shuffle(agents_ids)

		for agent_id in agents_ids:
			
			# Unpack the values #
			new_locations = new_measurements[agent_id]["location"]
			new_values = new_measurements[agent_id]["values"]

			if self.measured_locations is None:
				# Initialize the data #
				self.measured_locations = new_locations.reshape(-1,2)
				self.measured_values = new_values.reshape(-1,1)
			else:
				self.measured_locations = np.vstack((self.measured_locations, new_locations.reshape(-1,2)))
				self.measured_values = np.vstack((self.measured_values, new_values.reshape(-1,1)))

			# Fit the Gaussian Process #
			self.measured_locations, unique_index = np.unique(self.measured_locations, axis=0, return_index=True)
			self.measured_values = self.measured_values[unique_index]
			self.model.fit(self.measured_locations, self.measured_values.reshape(-1,1))
			# Compute the mean and the std values for all the visitable positions #
			new_mu_array, new_sigma_array = self.model.predict(self.visitable_positions[:], return_std=True)

			# Clip the mu to (0,1) #
			new_mu_array = np.clip(new_mu_array, 0.0, 1.0)
			new_sigma_array = np.clip(new_sigma_array**2, 1E-5, 999999.0)

			# Compute the intermediate metrics #
			unc_decrement = sigma_array - new_sigma_array
			model_change = mu_array - new_mu_array
			kl_div = np.abs(np.log(sigma_array/new_sigma_array) + 0.5 * (sigma_array**2 + (mu_array - new_mu_array)**2)/(new_sigma_array**2) - 0.5)
			# Store in the dictionary for later use #
			intermediate_model_metrics[agent_id] = {"kl": kl_div, "unc_decrement": unc_decrement, "model_change": model_change}

			mu_array = new_mu_array
			sigma_array = new_sigma_array


		# Conform the map of the surrogate model for the state with the final update #
		self.mu[self.visitable_positions[:, 0],
				self.visitable_positions[:, 1]] = mu_array

		self.sigma[self.visitable_positions[:, 0],
					self.visitable_positions[:, 1]] = sigma_array

		self.intermediate_model_metrics = intermediate_model_metrics

		
	def reset(self):
		""" Reset the environment """

		# Reset the fleet #
		self.fleet.reset_fleet()
		# Reset the benchmark#
		self.benchmark.reset()
		# Take new measurements #
		new_measurements = self.take_measurements()
		# Update the model
		self.measured_values = None
		self.measured_locations = None
		self.intermediate_models = None
		self.update_model(new_measurements=new_measurements)
		# Generate new state #
		self.state = self.generate_states()

		self.collisions = 0

		return self.state

	
	def take_measurements(self):
		""" This method conform a measurement for the fleet """

		new_measurements = {i:{"location": None, "values": None } for i in range(self.number_of_agents)}
		for id_vehicle, vehicle in enumerate(self.fleet.vehicles):
			# Extract the measurements #
			value = self.benchmark.read(position = vehicle.position)
			new_measurements[id_vehicle]["location"] = np.atleast_2d(vehicle.position)
			new_measurements[id_vehicle]["values"] = np.atleast_2d(value)

		return new_measurements
 
	
	def step(self, actions: dict):
		""" Process the actions. """

		# Compute the movement #
		if self.movement_type == "DISCRETE":
			movements = {i: (2*np.pi/self.number_of_movements * actions[i], self.movement_limits[0]) for i in actions.keys()}
		else:
			movements = {i: (0.5*(actions[i][0] + 1) * 2 * np.pi, self.movement_limits[0]) for i in actions.keys()}

		# Move the fleet #
		result = self.fleet.move_fleet_sincro(movements)
		self.collisions += np.sum([1 if res == "COLLISION" else 0 for res in result])

		# Take new measurements #
		new_measurements = self.take_measurements()

		# Update the model #
		self.update_model(new_measurements=new_measurements)

		# Compute reward #
		reward = self.reward_function(result)

		# Compute new state #
		self.state = self.generate_states()

		# Compute termination condition #
		dones = {i: False for i in range(self.number_of_agents)}
		for veh_id, vehicle in enumerate(self.fleet.vehicles):
			dones[veh_id] = vehicle.distance > vehicle.max_travel_distance or self.collisions > self.max_collisions
			self.active_agents[veh_id] = dones[veh_id]
		 
		return self.state, reward, dones, {}

	
	def reward_function(self, movement_result):

		# Depending on the reward function, chose the appropi #
		if self.reward_type == 'KL':
			reward = {i: np.mean(np.tanh(metric['kl'])) for i, metric in self.intermediate_model_metrics.items() if i in movement_result.keys()}
		elif self.reward_type == 'Uncertainty':
			reward = {i: np.sum(metric['unc_decrement']) for i, metric in self.intermediate_model_metrics.items() if i in movement_result.keys()}
		elif self.reward_type == 'Change':
			reward = {i: np.sum(np.abs(metric['model_change'])) for i, metric in self.intermediate_model_metrics.items() if i in movement_result.keys()}
		else:
			raise NotImplementedError("Reward function nor implemented!")

		# Penalize if necesary
		for agent_id, result in movement_result.items():
			reward[agent_id] = -1.0 if result == "COLLISION" else reward[agent_id]

		return reward

	
	def generate_states(self):
		""" Compute the state """

		if self.reward_type == 'Uncertainty':
			states = {i: np.concatenate((self.navigation_map[np.newaxis],
										self.sigma[np.newaxis],
										self.fleet.compute_fleet_map(vehicles = i)[np.newaxis],
										self.fleet.compute_fleet_map(vehicles = [j for j in range(self.number_of_agents) if j != i])[np.newaxis]
			))
						for i in range(self.number_of_agents) if self.active_agents[i]}
		
		else:
			states = {i: np.concatenate((self.navigation_map[np.newaxis],
										self.sigma[np.newaxis],
										self.mu[np.newaxis],
										self.fleet.compute_fleet_map(vehicles = i)[np.newaxis],
										self.fleet.compute_fleet_map(vehicles = [j for j in range(self.number_of_agents) if j != i])[np.newaxis]
			))
						for i in range(self.number_of_agents) if self.active_agents[i]}			



		return 

	
	def render(self):

		if self.fig is None:

			self.fig, self.axs = plt.subplots(2,3)

			self.d0 = self.axs[0,0].imshow(self.navigation_map, cmap = 'gray')
			self.d1 = self.axs[0,1].imshow(self.mu, vmax = 1.0, cmap='jet')
			self.d2 = self.axs[0,2].imshow(self.sigma, cmap='gray')

			self.d3 = self.axs[1,0].imshow(self.fleet.compute_fleet_map(), cmap = 'gray')
			self.d4 = self.axs[1,1].imshow(self.fleet.compute_fleet_map(vehicles=[1]), cmap='gray')
			self.d5 = self.axs[1,2].imshow(self.benchmark.ground_truth_field, cmap='jet')

		else:

			self.d1.set_data(self.mu)
			self.d2.set_data(self.sigma)
			self.d3.set_data(self.fleet.compute_fleet_map())
			self.d4.set_data(self.fleet.compute_fleet_map([1]))
			self.d5.set_data(self.benchmark.ground_truth_field)

		self.fig.canvas.draw()

		plt.pause(0.1)

		
	def get_action_mask(self, ind: int) -> np.ndarray:
		""" Get the invalid action mask for a certain vehicle (*ind*)

		:param ind: The index of the vehicle

		:return:
			- mask: The invalid action mask for the vehicle, where true means that the action is valid.

		"""

		angles = 2 * np.pi * np.arange(0, self.number_of_movements) / self.number_of_movements

		possible_points = self.fleet.vehicles[ind].position + 1.05 * self.movement_limits[0] * np.column_stack((np.cos(angles), np.sin(angles)))

		return np.asarray(list(map(self.fleet.vehicles[ind].is_the_position_valid, possible_points)))
	

	def get_safe_action(self, ind: int):

		action_mask = self.get_action_mask(ind=ind)

		return np.random.choice(np.arange(0,len(action_mask)), p=action_mask.astype(float)/action_mask.sum())

	

if __name__ == "__main__":

	from Environment.GroundTruths.ShekelGroundTruth import Shekel

	N = 4
	nav_map = np.genfromtxt('Environment/NavigationMaps/ypacarai_map.txt')
	gt_config = Shekel.sim_config_template
	gt_config["navigation_map"] = nav_map
	gt = Shekel(gt_config)

	env = StaticIGEnv(number_of_vehicles = N,
				 		navigation_map = nav_map,
				 		max_travel_distance = 100,
				 		initial_vehicle_positions = np.array([[20,20],[25,25],[30,30],[10,10]]),
				 		movement_type = "DISCRETE",
						movement_limits = (3,8),
						benchmark = gt,
						max_collisions = 10,
						# Model parameters #
						kernel_type = "Matern",
						lengthscale = 8.5,
						# Reward style #
						reward_type = "KL",
						# Default parameters #
						number_of_movements = 8,
				 )
	
	# Test a random agent #
	env.reset()
	done = {i: False for i in range(N)}
	actions = {i:env.get_safe_action(i) for i in range(N)}
	
	while not any(list(done.values())):

		env.render()

		for i in range(N):
			if not env.get_action_mask(i)[actions[i]]:
				actions[i] = env.get_safe_action(i)
				
		s,r,done,i = env.step(actions)
		print(done)
		print(r)

		env.render()



