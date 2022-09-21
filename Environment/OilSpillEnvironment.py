import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from groundtruth import GroundTruth


class OilSpill(GroundTruth):

	sim_config_template = {
		"navigation_map": np.ones((100, 100)),
		"seed": 9875123,
		"dt": 1.0,
		"kw": 1.0,
		"kc": 1.0,
		"gamma": 1.0,
		"flow": 10.0,
		'normalize': True,
		'initial_time': 10,
	}

	def __init__(self, sim_config: dict):
		super().__init__(sim_config)

		self.seed = sim_config['seed']
		self.initial_time = sim_config["initial_time"]

		np.random.seed(self.seed)

		self.done = None
		self.im0 = None
		self.im1 = None
		self.v = None
		self.u = None

		self.boundaries_map = sim_config['navigation_map']
		self.x, self.y = np.meshgrid(np.arange(0, self.boundaries_map.shape[0]),
		                             np.arange(0, self.boundaries_map.shape[1]))

		self.visitable_positions = np.column_stack(np.where(self.boundaries_map == 1)).astype(float)

		self.dt = sim_config['dt']
		self.Kw = sim_config['kw']
		self.Kc = sim_config['kc']
		self.gamma = sim_config['gamma']

		self.x_bins = np.arange(0, self.boundaries_map.shape[0]+1)
		self.y_bins = np.arange(0, self.boundaries_map.shape[1]+1)

		# Environment variables
		self.source_points = None
		self.wind_speed = None
		self.source_fuel = None
		self.contamination_position = None
		self.contamination_speed = None
		self.ground_truth_field = None
		self.flow = int(sim_config['flow'])
		self.spill_directions = None

		self.particles_speeds = None

		self.init = False

		self.ax = None
		self.fig = None

	def reset(self, random_benchmark=True):
		"""Reset the env variables"""

		if not self.init:
			self.init = True

		self.done = False

		if random_benchmark:
			self.seed += 1

		random_indx = np.random.RandomState(self.seed).choice(np.arange(0,len(self.visitable_positions)), np.random.RandomState(self.seed).randint(1,3), replace=False)
		self.source_points = np.copy(self.visitable_positions[random_indx])
		self.wind_speed = np.random.RandomState(self.seed).rand(2) * 2 - 1
		self.source_fuel = 10000
		self.contamination_position = np.copy(self.source_points)

		x0 = np.random.RandomState(self.seed).randint(0, self.boundaries_map.shape[0])
		y0 = np.random.RandomState(self.seed).randint(0, self.boundaries_map.shape[1])

		# Current vector field
		self.u = np.sin(np.pi * (self.x - x0) / 50) * np.cos(np.pi * (self.y - y0) / 50)
		self.v = -np.cos(np.pi * (self.x - x0) / 50) * np.sin(np.pi * (self.y - y0) / 50)

		# Density map
		self.ground_truth_field = np.zeros_like(self.boundaries_map)

		if self.initial_time is not None:
			self.update_to_time(self.initial_time)

	def step(self):
		""" Process the action and update the environment state """

		assert self.init, "Environment not initiated!"

		# Generate new particles
		for source_point in self.source_points:
			# While there is enough fuel
			if self.source_fuel > 0:
				for _ in range(self.flow):
					
					# Compute the components of the particle movement #
					v_random = self.gamma * (np.random.rand(2) * 2 - 1)
					v_wind = self.Kw * self.wind_speed
					v_current = self.Kc * self.get_current_speed(source_point)
					# Add the new position to the list #
					v_new = source_point + self.dt * (v_wind + v_current) + v_random

					if self.boundaries_map[v_new[0].astype(int),v_new[1].astype(int)] == 0:
						v_new = np.copy(source_point)

					self.contamination_position = np.vstack((self.contamination_position, v_new))
					self.source_fuel -= 1
		
		# Update the particles positions #
		expelled = []
		for i in range(len(self.contamination_position)):

			# Compute the components of the particle movement #
			v_random = self.gamma * (np.random.rand(2) * 2 - 1)
			v_wind = self.Kw * self.wind_speed
			v_current = self.Kc * self.get_current_speed(self.contamination_position[i])
			# Add the new position to the list #
			v_new = self.dt * (v_wind + v_current) + v_random
			new_position = self.contamination_position[i, :] + v_new
			new_position = np.clip(new_position, 0, (self.boundaries_map.shape[0]-1, self.boundaries_map.shape[1]-1))
			# Update the positions #
			if self.boundaries_map[new_position[0].astype(int), new_position[1].astype(int)] == 1:
				if self.ground_truth_field[new_position[0].astype(int), new_position[1].astype(int)] < 50:
					self.contamination_position[i, :] = new_position

			if any(self.contamination_position[i] > self.boundaries_map.shape) or any(self.contamination_position[i] < 0):
				expelled.append(i)

		self.contamination_position = np.delete(self.contamination_position, expelled, axis=0)
		self.ground_truth_field, _, _ = np.histogram2d(self.contamination_position[:, 0], self.contamination_position[:, 1],
		                                               [self.x_bins, self.y_bins])

		self.ground_truth_field = gaussian_filter(self.ground_truth_field / 10.0, 1.0)

		return self.ground_truth_field

	def get_current_speed(self, position):
		int_position = position.astype(int)
		return np.array([self.v[int_position[1], int_position[0]], self.u[int_position[1], int_position[0]]])

	def render(self):

		if self.im1 is None:

			self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
			self.ax[0].quiver(self.x, self.y, self.u, self.v, scale=100)
			self.ax[0].set_xlim((0, self.boundaries_map.shape[0]))
			self.ax[0].set_ylim((0, self.boundaries_map.shape[1]))
			self.ax[0].invert_yaxis()
			self.im0 = self.ax[0].scatter(self.contamination_position[:, 0], self.contamination_position[:, 1])
			rendered = np.copy(self.boundaries_map) * self.ground_truth_field
			rendered[self.boundaries_map == 0] = np.nan
			self.im1 = self.ax[1].imshow(rendered.T, interpolation=None, cmap='jet', vmin=0, vmax=3)

		else:
			rendered = np.copy(self.boundaries_map) * self.ground_truth_field
			rendered[self.boundaries_map == 0] = np.nan
			self.im0.set_offsets(self.contamination_position)
			self.im1.set_data(rendered.T)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		plt.pause(0.0001)

	def read(self, position=None):
		""" Read the complete ground truth or a certain position """

		if position is None:
			return self.ground_truth_field
		else:
			return self.ground_truth_field[position[0]][position[1]]

	def update_to_time(self, t):

		for _ in range(t):
			self.step()

		return self.ground_truth_field


if __name__ == '__main__':

	my_map = np.ones((100,100))

	config_dict = OilSpill.sim_config_template
	config_dict['navigation_map'] = my_map

	env = OilSpill(config_dict)
	env.reset()

	for _ in range(10):
		for _ in range(100):
			env.step()
			env.render()
		env.reset(random_benchmark=False)

	plt.show()
