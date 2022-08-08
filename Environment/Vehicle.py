from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, unique

@unique
class FleetState(Enum):

	WAITING_FOR_ACTION = 0
	COLLIDED = 1
	ON_WAY = 2
	GOAL_REACHED = 3
	LAST_ACTION = 4
	FINISHED = 5


class Vehicle:

	def __init__(self, vehicle_id: Union[int, str], initial_position: np.ndarray, vehicle_configuration: dict):

		""" This class defines a vehicle in a 2D plane. The vehicle can move and take samples according to a
		Ground Truth. The vehicle receives a new position and tries to reach it. If it is not possible,
		a flag is raises.  Once the movement is completed, a sample is taken. """

		self.vehicle_configuration = vehicle_configuration
		self.initial_position = initial_position
		self.position = self.initial_position.copy()
		self.dt = vehicle_configuration['dt']
		# The navigation map as a binary matrix. The vehicle can move only on the free cells.
		self.navigation_map = vehicle_configuration['navigation_map']
		# Navigation map threshold #
		self.target_threshold = vehicle_configuration['target_threshold']
		# Ground truth #
		self.ground_truth = vehicle_configuration['ground_truth']
		self.measurement_size = vehicle_configuration['measurement_size']
		# The vehicle id #
		self.agent_id = vehicle_id
		self.vehicle_state = FleetState.WAITING_FOR_ACTION
		self.target_position = None
		# The travelled distance #
		self.distance = 0
		self.max_travel_distance = vehicle_configuration['max_travel_distance']


	def reset_vehicle(self):
		""" Reset the vehicle position and return the first sample """
		self.position = self.initial_position.copy()
		self.vehicle_state = FleetState.WAITING_FOR_ACTION
		self.target_position = self.position.copy()
		self.distance = 0
		# Take the initial measurement #
		return self.take_measurement()

	def move_to(self, attemped_position: np.ndarray) -> [bool, np.ndarray]:

		""" This function moves the vehicle to a new position. If the movement is possible, the function returns
		True. If the movement is not possible, the function returns False. The funtion also returns a sample value """
		original_position = self.position.copy()

		# Moves to the attempted position with intervals of dt until an invalid position of navigation map is reached #
		valid = True
		# The direction vector from the current position and the target #
		direction_vector = (attemped_position - self.position) / np.linalg.norm(attemped_position - self.position)
		while valid:
			# Increment the attempted position #
			valid = self.step_with_direction(direction_vector)
			# If the position is sufficiently near the target, finish the movement #
			if np.linalg.norm(self.position - attemped_position) < self.target_threshold:
				break

		# Update the vehicle position #
		self.distance += np.linalg.norm(original_position - self.position)
		# Return the sample #
		return valid, self.take_measurement()

	def set_target(self, target_position):
		""" Set the target position. Put the vehicle state to ON_WAY """
		assert self.vehicle_state not in [FleetState.FINISHED, FleetState.LAST_ACTION], "The vehicle cannot move anymore!"
		self.target_position = target_position
		self.vehicle_state = FleetState.ON_WAY

	def step(self):
		""" Update the state of the vehicle """

		original_position = self.position.copy()

		# If the vehicle is in a collision state, return to WAITING_FOR_ACTION #
		if self.vehicle_state != FleetState.LAST_ACTION:

			if self.vehicle_state == FleetState.COLLIDED:
				self.vehicle_state = FleetState.WAITING_FOR_ACTION
				self.target_position = self.position.copy()

			# If a position is targeted, then update the position #
			elif self.vehicle_state == FleetState.ON_WAY:
				direction_vector = (self.target_position - self.position) / np.linalg.norm(self.target_position - self.position + 1e-6)
				is_valid = self.step_with_direction(direction_vector)
				# Update the state depending on the validity of the step
				self.vehicle_state = FleetState.ON_WAY if is_valid else FleetState.COLLIDED
				# Check if the target has been reached and change the state to WAITING #
				if np.linalg.norm(self.position - self.target_position) < self.target_threshold:
					self.vehicle_state = FleetState.WAITING_FOR_ACTION


			# Update the distance #
			self.distance += np.linalg.norm(original_position - self.position)

			# If the maximum distance is reached, then change the state to FINISHED #

			if self.distance >= self.max_travel_distance:
				self.vehicle_state = FleetState.LAST_ACTION

		# Take a measurement if reached #
		measurement = self.take_measurement() if self.vehicle_state in [FleetState.WAITING_FOR_ACTION, FleetState.COLLIDED, FleetState.LAST_ACTION] else None

		return self.vehicle_state, measurement


	def step_with_direction(self, direction):
		""" This function moves the vehicle in a direction. If the movement is possible, the function returns
		True. If the movement is not possible, the function returns False. """

		next_position = self.position + direction * self.dt
		valid = self.is_the_position_valid(next_position)
		if valid:
			self.position = next_position
			return True
		else:
			return False

	def is_the_position_valid(self, position):
		""" This function checks if the position is valid """

		out_bounds_condition = position[0] < 0 or position[0] >= self.navigation_map.shape[0] or position[1] < 0 or position[1] >= self.navigation_map.shape[1]

		lower_bound = np.floor(position).astype(int)
		upper_bound = np.ceil(position).astype(int)


		collide_condition = self.navigation_map[lower_bound[0], lower_bound[1]] == 0 or \
		                    self.navigation_map[upper_bound[0], upper_bound[1]] == 0 or \
		                    self.navigation_map[lower_bound[0], upper_bound[1]] == 0 or \
							self.navigation_map[upper_bound[0], lower_bound[1]] == 0

		if out_bounds_condition or collide_condition:
			return False
		else:
			return True

	def take_measurement(self):

		upp_lims = np.clip(self.position + self.measurement_size, 0, self.navigation_map.shape).astype(int)
		low_lims = np.clip(self.position - self.measurement_size, 0, self.navigation_map.shape).astype(int)

		measurement_mask = self.ground_truth[low_lims[0]: upp_lims[0] + 1, upp_lims[1]: upp_lims[1] + 1]

		return {'data': measurement_mask, 'position': self.position}


if __name__ == '__main__':

	plt.ion()

	""" Test the single agent """
	navigation_map = np.genfromtxt('./wesslinger_map.txt')

	plt.matshow(navigation_map)

	vehicle_configuration = {
		'dt': 0.5,
		'navigation_map': navigation_map,
		'target_threshold': 0.5,
		'ground_truth': np.random.rand(50,50),
		'measurement_size': np.array([4, 4])
	}

	vehicle = Vehicle(1, initial_position=np.array([15, 19]), vehicle_configuration = vehicle_configuration)
	vehicle.reset_vehicle()
	plt.plot(vehicle.position[0], vehicle.position[1], 'ro')
	plt.pause(1)

	for i in range(10):
		target = np.random.rand(2) * navigation_map.shape
		plt.plot(target[1], target[0], 'gx')
		vehicle.move_to(target)
		plt.plot(vehicle.position[1], vehicle.position[0], 'ro')
		plt.pause(1)
		plt.show()
		print(vehicle.position)
		print(vehicle.distance)
		print(vehicle.take_measurement())
		print('\n')
