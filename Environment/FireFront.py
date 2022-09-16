import math
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from groundtruth import GroundTruth
from collections import defaultdict

FIRE = 0
FUEL = 1


class WildfireSimulator(GroundTruth):

	sim_config_template = {
		"navigation_map": np.ones((50, 50)),
		"width": 100,
		"height": 100,
		"init_fire_mode": "central",  # Options random, central
		"init_hotspots": 2,  # Select the number of hotspots
		"seed": 9875123,
		"max_ignition_distance": 5,
		"ignition_factor": 0.005/2,
		"wind_type": 'random',  # Options None, user, random
		"wind_speed": 0,
		"wind_angle": -20,
		"initial_time": 100,
	}

	def __init__(self, sim_config: dict):
		super().__init__(sim_config)

		self.sim_config = sim_config
		self.navigation_map = sim_config['navigation_map']
		self.width = sim_config["width"]
		self.height = sim_config["height"]
		self.init_fire_mode = sim_config["init_fire_mode"]
		self.initial_time = sim_config["initial_time"]

		if sim_config["seed"] is None:
			self.seed = time.time_ns()
		else:
			self.seed = sim_config["seed"]

		self.max_distance = sim_config["max_ignition_distance"]
		self.ignition_factor = sim_config["ignition_factor"]

		self.burning_map = np.zeros((self.width, self.height))
		self.belief_map = np.zeros((self.width, self.height))
		self.fuel_map = np.full((self.width, self.height), 25)

		self.wind_angle = None
		self.wind_speed = None
		self.wind_type = sim_config["wind_type"]

		self.burning_dict = {}  # Key is cell position [x,y], value [Fire boolean, fuel integer]
		self.ignite_dict = defaultdict(lambda: 1)
		self.wind = None

		self.ax, self.fig, self.drawing = None, None, None

		assert (self.height > self.navigation_map.shape[0] \
		       and self.width > self.navigation_map.shape[1]), "The navigation map should be smaller than the base map for fire calculation"

		self.x_bins = np.linspace(0, self.height, self.navigation_map.shape[0]+1)
		self.y_bins = np.linspace(0, self.width, self.navigation_map.shape[1]+1)
		self.ground_truth_field = None

		random.seed(self.seed)

	def init_fire(self):

		if self.init_fire_mode == "random":
			x, y = random.randrange(0, self.width - 1), random.randrange(0, self.height - 1)
		elif self.init_fire_mode == "central":
			x, y = self.width // 2, self.height // 2
		else:
			raise NotImplementedError("This init. setting for the fire is not implemented yet.")

		self.burning_dict[(x, y)] = [True, self.fuel_map[x][y], True]
		self.burning_map[x][y] = 1

	def reset(self, random_gt: bool = False):

		self.burning_map = np.zeros((self.width, self.height))
		self.belief_map = np.zeros((self.width, self.height))
		self.fuel_map = np.full((self.width, self.height), 15)

		self.wind_angle = 0 if self.wind_type is None \
			else np.random.rand() * 2 * np.pi if self.wind_type == 'random' \
			else np.pi * self.sim_config["wind_angle"] / 180.0

		self.wind_speed = 0 if self.wind_type is None \
			else np.random.rand() * 3 if self.wind_type == 'random' \
			else self.sim_config["wind_speed"]

		self.burning_dict = {}  # Key is cell position [x,y], value [Fire boolean, fuel integer]
		self.ignite_dict = defaultdict(lambda: 1)

		self.wind = (self.wind_speed * math.cos(self.wind_angle), self.wind_speed * math.sin(self.wind_angle))

		self.init_fire()

		fire_field = np.column_stack(np.where(self.burning_map == 1))

		self.ground_truth_field, _, _ = np.histogram2d(fire_field[:, 0], fire_field[:, 1], [self.x_bins, self.y_bins])
		self.ground_truth_field = self.ground_truth_field * self.navigation_map

		if self.initial_time is not None:
			self.update_to_time(self.initial_time)

	def step(self):

		extinguished_cells = []
		counter = 0

		for position in self.burning_dict:

			fuel_contact = 0
			# Update ignition probabilities by calculating the probability of igniting the surroundings.
			xk0 = max(0, math.floor(position[0] - self.max_distance))
			xk1 = min(self.width, math.ceil(position[0] + self.max_distance))

			yk0 = max(0, math.floor(position[1] - self.max_distance))
			yk1 = min(self.height, math.ceil(position[1] + self.max_distance))

			if self.burning_dict[position][2]:

				counter += 1
				for xk in range(xk0, xk1):
					for yk in range(yk0, yk1):
						if self.burning_map[xk][yk] == 0:
							dx = xk - position[0]
							dy = yk - position[1]
							dist2 = dx ** 2 + dy ** 2
							if dist2 and dist2 <= self.max_distance ** 2:
								wind_factor = dx * self.wind[0] + dy * self.wind[1]
								p = max(0, min(1, self.ignition_factor * (1 + wind_factor) / dist2))
								# p = 0.05
								self.ignite_dict[(xk, yk)] = self.ignite_dict[(xk, yk)] * (1 - p)
							fuel_contact = 1

			if fuel_contact == 0:
				self.burning_dict[position][2] = False

			# Update fuel
			if self.burning_dict[position][FUEL] > 0:
				self.burning_dict[position][FUEL] -= 1
				self.fuel_map[position[0]][position[1]] -= 1
			else:
				extinguished_cells.append(position)
				self.burning_map[position[0]][position[1]] = -1

		# Clean up burning dictionary
		for cell in extinguished_cells:
			self.burning_dict.pop(cell)

		# Ignite new cells
		remove_ignited = []
		for cell in list(self.ignite_dict.keys()):
			if random.random() > self.ignite_dict[cell]:
				self.burning_dict[cell] = [True, self.fuel_map[cell[0]][1], True]
				self.burning_map[cell[0]][cell[1]] = 1
				remove_ignited.append(cell)

		for cell in remove_ignited:
			self.ignite_dict.pop(cell)

		fire_field = np.column_stack(np.where(self.burning_map == 1))
		self.ground_truth_field, _, _ = np.histogram2d(fire_field[:, 0], fire_field[:, 1], [self.x_bins, self.y_bins])
		self.ground_truth_field = gaussian_filter(self.ground_truth_field, 1.0)*self.navigation_map/3

		return self.ground_truth_field

	def render(self):

		if self.fig is None:
			self.fig, self.ax = plt.subplots(1, 1)
			self.drawing = self.ax.imshow(self.ground_truth_field, vmin=0, vmax=5, cmap='hot')
		else:
			self.drawing.set_data(self.ground_truth_field)

		self.fig.canvas.draw()
		plt.pause(0.01)
		plt.draw()

	def update_to_time(self, t):

		for _ in range(t):
			self.step()

		return self.ground_truth_field

	def read(self, pos=None):

		if pos is None:
			return self.ground_truth_field
		else:
			return self.ground_truth_field[pos[0], pos[1]]


if __name__ == '__main__':

	sim_config = WildfireSimulator.sim_config_template
	sim_config['navigation_map'] = np.genfromtxt('./wesslinger_map.txt')

	sim = WildfireSimulator(sim_config)
	before = time.process_time()

	for _ in range(20):
		sim.reset()
		for i in range(30):
			sim.step()
			sim.render()
		print(time.process_time() - before)


# plt.imshow(sim.burning_map)
# plt.draw()
# plt.pause(0.001)
# print(sim.burning_dict)
# print(sim.ignite_dict)