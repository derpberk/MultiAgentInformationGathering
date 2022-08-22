import numpy as np
from Vehicle import Vehicle, FleetState
import matplotlib.pyplot as plt


class Fleet:

	def __init__(self, fleet_configuration: dict):
		""" A fleet coordinator object that is responsible for the coordination of the vehicles in the fleet.
		It is in charge of dispatching the reset conditions, process the movements and the measurements of every agent.

		Args:
			fleet_configuration (dict): A dictionary containing the configuration of the fleet.
		"""

		self.fleet_configuration = fleet_configuration
		self.number_of_vehicles = fleet_configuration['number_of_vehicles']

		# Create a bunch of vehicles
		self.vehicles = [Vehicle(vehicle_id=i,
		                         initial_position=fleet_configuration['initial_positions'][i],
		                         vehicle_configuration=fleet_configuration['vehicle_configuration']) for i in range(self.number_of_vehicles)]

		# Fleet state and measurements #
		# NOTE: The fleet state is a vector of the states of the vehicles. It is conformed to define if the vehicle is moving, has collided, or
		# finished the mission. This atribute is conformed in the fleet object using each vehicle information (individual conditions) or fleet info
		# (colective conditions) such as in between individual collitions #
		self.fleet_state = [None for _ in range(self.number_of_vehicles)]
		self.measurements = [None for _ in range(self.number_of_vehicles)]

	def reset(self):
		""" Reset the fleet position and measurements.

		 Returns:
		 	fleet_state: List of states of the fleet
		"""

		self.measurements = [vehicle.reset_vehicle() for vehicle in self.vehicles]
		self.fleet_state = [vehicle.vehicle_state for vehicle in self.vehicles]

		return self.measurements


	def set_target_position(self, agent_id, target_position):
		""" Set a new position target for an agent if it is not finished """

		assert self.fleet_state[agent_id] is not FleetState.FINISHED, "The vehicle has finished and cannot move anymore!"

		self.fleet_state[agent_id] = self.vehicles[agent_id].set_target(target_position)  # Set the new position and update the state #

	def step(self) -> (list, list):
		""" Compute one discrete step for every vehicle and retreive the measurements and new states if they are available.
		 Note that when the vehicle has finished the mission or is moving, the measurement is None. """

		for vehicle_id, vehicle in enumerate(self.vehicles):
			# Only update the vehicle if it is not finished and not collided #
			if self.fleet_state[vehicle_id] not in [FleetState.FINISHED, FleetState.COLLIDED]:
				self.fleet_state[vehicle_id], self.measurements[vehicle_id] = vehicle.step()

		return self.fleet_state, self.measurements

	def update_syncronously(self) -> (list, list):
		""" Update the state of the fleet until all vehicles are not in ON_WAY state """

		# Iterare for every vehicle #
		for vehicle_id, vehicle in enumerate(self.vehicles):
			# IF the vehicle is in LAST_ACTION state, set the state to FINISHED #
			if vehicle.vehicle_state == FleetState.LAST_ACTION:
				vehicle.vehicle_state = FleetState.FINISHED
				self.fleet_state[vehicle_id] = FleetState.FINISHED

		# Process state until all vehicles are not in ON_WAY state, which is COLLIDED, OR LAST_ACTION #
		while any(s == FleetState.ON_WAY for s in self.fleet_state):
			self.step()

		return self.fleet_state, self.measurements

	def update_asyncronously(self):
		"""
		Update the state of the fleet until at least one vehicle is not in ON_WAY state

		Returns:
			fleet_state: List of states of the fleet
			measurements: List of measurements of the fleet

		"""

		# Iterare for every vehicle #
		for vehicle_id, vehicle in enumerate(self.vehicles):
			# IF the vehicle is in LAST_ACTION state, set the state to FINISHED #
			if vehicle.vehicle_state == FleetState.LAST_ACTION:
				vehicle.vehicle_state = FleetState.FINISHED
				self.fleet_state[vehicle_id] = FleetState.FINISHED

		# Take steps until at least one vehicle is not in ON_WAY state and this state is not LAST_ACTION #
		while all(s in [FleetState.LAST_ACTION, FleetState.FINISHED, FleetState.ON_WAY, FleetState.COLLIDED] for s in self.fleet_state):
			self.step()
			# If every vehicle is in LAST_ACTION state or has finished, stop stepping as the mission is over #
			if all(s in [FleetState.FINISHED, FleetState.LAST_ACTION, FleetState.COLLIDED] for s in self.fleet_state):
				break

		return self.fleet_state, self.measurements

	def get_positions(self) -> np.ndarray:
		"""
		Method to obtain the positions of the vehicles in the fleet.

		Returns:
			positions: numpy array of positions of the vehicles in the fleet.

		"""
		return np.asarray([vehicle.position for vehicle in self.vehicles])

	def set_state(self, agent_id, veh_state):
		""" Setter of a state of given agent """

		assert agent_id in [veh.agent_id for veh in self.vehicles], "Agent id {} is not in the fleet".format(agent_id)
		assert veh_state in FleetState, "State {} is not valid".format(veh_state)

		self.vehicles[agent_id].vehicle_state = veh_state
		self.fleet_state[agent_id] = veh_state



