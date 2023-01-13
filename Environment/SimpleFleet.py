import numpy as np
from Environment.SimpleVehicle import Vehicle
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


class Fleet:

    def __init__(self, 
                number_of_agents: int,
                navigation_map: np.ndarray,
                max_travel_distance: float,
                initial_vehicle_positions: np.ndarray,
                agent_size: int = 1
                ) -> None:

        # Create the vehicles #
        self.vehicles = [Vehicle(initial_position=initial_vehicle_positions[i],
                                 max_travel_distance=max_travel_distance,
                                 navigation_map=navigation_map,
                                 vehicle_id=i,
                                 agent_size=agent_size) for i in range(number_of_agents)]

        
        self.navigation_map = navigation_map
        self.number_of_vehicles = number_of_agents

        # Reset every agent position #
        self.reset_fleet()

    
    def reset_fleet(self):
        """ Reset every vehicle """
        
        for vehicle in self.vehicles:
            vehicle.reset_vehicle()


    def move_fleet_sincro(self, movements: dict):
        """ With this movement, every vehicle moves and waits for each other. """

        result = {i: self.vehicles[i].move_vehicle(movement[0], movement[1]) for i , movement in movements.items()}

        # Check collisions between agents #
        distance_mat = distance_matrix(self.fleet_position, self.fleet_position)
        distances = distance_mat[~np.eye(distance_mat.shape[0],dtype=bool)].reshape(distance_mat.shape[0],-1)
        
        for i in result.keys():

            if any(distances[i] < self.vehicles[i].agent_size):
                result[i] = 'COLLISION'
        
        return result

    def move_fleet_asincro(self, angles, distances):

        raise NotImplementedError("This method is not implemented yet!")

    def compute_fleet_map(self, vehicles=None):
        """ Compute the fleet map """

        if vehicles is None:
            vehicles = np.arange(0, self.number_of_vehicles)
        else:
            vehicles = np.atleast_1d(vehicles)

        fleet_map = np.zeros_like(self.navigation_map)

        for vehicle_id in vehicles:
            fleet_map[int(self.vehicles[vehicle_id].position[0]), int(self.vehicles[vehicle_id].position[1])] = 1.0

        return fleet_map

    @property
    def fleet_position(self):
        """ Return a 2D array with the positions of the drones """

        return np.asarray([vehicle.position.copy() for vehicle in self.vehicles])


    