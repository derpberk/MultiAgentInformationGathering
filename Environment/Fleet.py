import numpy as np
from Vehicle import Vehicle, FleetState
import matplotlib.pyplot as plt


class Fleet:

	def __init__(self, fleet_configuration: dict):
		self.fleet_configuration = fleet_configuration
		self.number_of_vehicles = fleet_configuration['number_of_vehicles']

		# Create a bunch of vehicles
		self.vehicles = [Vehicle(vehicle_id=i,
		                         initial_position=fleet_configuration['initial_positions'][i],
		                         vehicle_configuration=fleet_configuration['vehicle_configuration']) for i in range(self.number_of_vehicles)]

		# Fleet state #
		self.fleet_state = [None for _ in range(self.number_of_vehicles)]
		self.measurements = [None for _ in range(self.number_of_vehicles)]

	def reset(self):

		# Reset every vehicle #
		self.measurements = [vehicle.reset_vehicle() for vehicle in self.vehicles]
		self.fleet_state = [vehicle.vehicle_state for vehicle in self.vehicles]
		self.active_vehicles = [True for _ in range(self.number_of_vehicles)]

		return self.measurements


	def set_target_position(self, agent_id, target_position):
		""" Set the target position for the agents  with id agent_id"""
		assert self.fleet_state[agent_id] is not FleetState.FINISHED, "The vehicle has finished and cannot move anymore!"
		self.vehicles[agent_id].set_target(target_position)
		self.fleet_state[agent_id] = FleetState.ON_WAY

	def step(self):
		""" Compute one step for every vehicle and retreive the measurements and new states.
		 Note that when the vehiclehas finished or is moving, the measurement is None. """

		for vehicle_id, vehicle in enumerate(self.vehicles):

			# Only update the vehicle if it is not finished #
			if self.fleet_state[vehicle_id] != FleetState.FINISHED:
				self.fleet_state[vehicle_id], self.measurements[vehicle_id] = vehicle.step()

		return self.fleet_state, self.measurements

	def update_syncronously(self):
		""" Update the state of the fleet until all vehicles are not in ON_WAY state """

		# The vehicle state is FINISHED when the last updte was LAST_ACTION #
		for vehicle_id, vehicle in enumerate(self.vehicles):
			if vehicle.vehicle_state == FleetState.LAST_ACTION:
				vehicle.vehicle_state = FleetState.FINISHED
				self.fleet_state[vehicle_id] = FleetState.FINISHED

		while any(s == FleetState.ON_WAY for s in self.fleet_state):
			self.step()

		return self.fleet_state, self.measurements

	def update_asyncronously(self):
		""" Update the state of the fleet until at least one vehicle is not in ON_WAY state """

		# The vehicle state is FINISHED when the last updte was LAST_ACTION #
		for vehicle_id, vehicle in enumerate(self.vehicles):
			if vehicle.vehicle_state == FleetState.LAST_ACTION:
				vehicle.vehicle_state = FleetState.FINISHED
				self.fleet_state[vehicle_id] = FleetState.FINISHED

		# Take steps until at least one vehicle is not in ON_WAY state and this state is not LAST_ACTION #
		while all(s in [FleetState.LAST_ACTION, FleetState.FINISHED, FleetState.ON_WAY] for s in self.fleet_state):
			self.step()
			if all(s in [FleetState.FINISHED, FleetState.LAST_ACTION] for s in self.fleet_state):
				break

		return self.fleet_state, self.measurements

	def get_positions(self):

		return np.asarray([vehicle.position for vehicle in self.vehicles])

	def set_state(self, agent_id, veh_state):
		""" Set state of given agent """
		assert agent_id in [veh.agent_id for veh in self.vehicles], "Agent id {} is not in the fleet".format(agent_id)
		assert veh_state in FleetState, "State {} is not valid".format(state)

		self.vehicles[agent_id].vehicle_state = veh_state
		self.fleet_state[agent_id] = veh_state

if __name__ == '__main__':

	plt.ion()

	navigation_map = np.genfromtxt('./wesslinger_map.txt')
	N = 4
	plt.imshow(navigation_map)
	plt.show()

	vehicle_configuration = {
		'dt': 0.5,
		'navigation_map': navigation_map,
		'target_threshold': 0.5,
		'ground_truth': np.random.rand(50,50),
		'measurement_size': np.array([4, 4])
	}


	fleet_configuration = {
		'vehicle_configuration': vehicle_configuration,
		'number_of_vehicles': N,
		'initial_positions': np.array([[15, 19],
		                                [13, 19],
		                                [18, 19],
		                                [15, 22]])
	}

	fleet = Fleet(fleet_configuration)
	fleet.reset()

	target_positions = fleet.get_positions() + np.array([0, 5])
	for i in range(N):
		fleet.set_target_position(i, target_positions[i])
		plt.plot(target_positions[i,1], target_positions[i,0], 'go')

	for t in range(100):

		state, meas = fleet.step()

		print("Fleet state ->  ", state)

		for i in range(N):
			plt.plot(fleet.vehicles[i].position[1], fleet.vehicles[i].position[0], 'rx')

		if all(s == FleetState.WAITING_FOR_ACTION for s in state):
			rnd = np.random.randint(0,7)
			target_positions = fleet.get_positions() + np.array([[0,1],[1,0],[1,1],[-1,1],[-1,-1],[-1,0],[0,-1],[1,-1]])[rnd] * 5
			for i in range(N):
				fleet.set_target_position(i, target_positions[i])
				plt.plot(target_positions[i, 1], target_positions[i, 0], 'go')

		plt.show()
		plt.pause(0.1)


