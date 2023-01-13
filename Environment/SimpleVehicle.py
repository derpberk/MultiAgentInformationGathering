import numpy as np
import matplotlib.pyplot as plt
from typing import Union

class Vehicle:
	""" Simple vehicle that moves. """

	def __init__(self, 
				vehicle_id: Union[int, str], 
				initial_position: np.ndarray,
				navigation_map: np.ndarray,
				max_travel_distance: float,
				agent_size = 1):

		# Set the initial position #
		self.initial_position = initial_position
		self.position = self.initial_position.copy()

		# The navigation map as a binary matrix. The vehicle can move only on the free cells.
		self.navigation_map = navigation_map

		# The vehicle id #
		self.agent_id = vehicle_id

		# The travelled distance #
		self.distance = 0
		self.max_travel_distance = max_travel_distance
		# The agent size
		self.agent_size = agent_size

		# Number of invalid movements
		self.collisions = 0
		self.wps = []

	def reset_vehicle(self):
		"Reset the vehicle position "

		self.position = self.initial_position.copy()
		self.wps.append(self.position.copy())

		# Reset also the distance #
		self.distance = 0.0
		self.collisions = 0

	
	def move_vehicle(self, angle: float, distance: float):
		""" Move the vehicle the given angle with a given distance.
		Also computes if there is a collision. """

		next_attempred_position = self.position + distance * np.array([np.cos(angle), np.sin(angle)])
		
		# Check if it is a valid movement #
		valid_movement = self.is_the_position_valid(next_attempred_position)

		if valid_movement:
			# If valid, accumulate the distance #
			self.distance += np.linalg.norm(next_attempred_position - self.position)
			# Update the position #
			self.position = next_attempred_position.copy()
			self.wps.append(self.position.copy())
			# Return a flag depending on the final condition #
			if self.distance >= self.max_travel_distance:
				return "DONE"
			else:
				return "OK"

		else:
			self.collisions += 1
			return "COLLISION"
		
	
	def is_the_position_valid(self, position):
		""" This function checks if the position is valid.
		"""

		# Clip the position into the map #
		position = np.clip(position, 0, np.asarray(self.navigation_map.shape) - 1)
		out_bounds_condition = position[0] < 0 or position[0] >= self.navigation_map.shape[0] or position[1] < 0 or position[1] >= self.navigation_map.shape[1]

		# Check all positions in the neighborhood of the target point. This is necessary due to the discretization of the map. #
		lower_bound = np.floor(position).astype(int)
		upper_bound = np.ceil(position).astype(int)

		# Condition for colliding
		if self.agent_size <= 1:
			collide_condition = self.navigation_map[lower_bound[0], lower_bound[1]] == 0 or self.navigation_map[upper_bound[0], upper_bound[1]] == 0 or \
								self.navigation_map[lower_bound[0], upper_bound[1]] == 0 or self.navigation_map[upper_bound[0], lower_bound[1]] == 0
		else:
			# Compute a circular neighborhood around the agent of size agent_size. If there is an obstacle, then flag up the collision
			collide_condition = self.compute_collision_condition(self.navigation_map, position, self.agent_size)

		if out_bounds_condition or collide_condition:
			return False
		else:
			return True

	@staticmethod
	def compute_collision_condition(navigation_map, position, size):
		""" Compute the circular mask """

		px, py = position.astype(int)

		# State - coverage area #
		x = np.arange(0, navigation_map.shape[0])
		y = np.arange(0, navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= size ** 2

		return any(navigation_map[mask.T] == 0)
