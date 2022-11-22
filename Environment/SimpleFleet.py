import numpy as np
from Environment.SimpleVehicle import Vehicle
import matplotlib.pyplot as plt


class Fleet:

    def __init__(self, 
                number_of_agents: int,
                navigation_map: np.ndarray,
                max_travel_distance: float,
                initial_vehicle_positions: np.ndarray,
                ) -> None:

        # Create the vehicles #
        self.vehicles = [Vehicle(initial_position=initial_vehicle_positions[i],
                                 max_travel_distance=max_travel_distance,
                                 navigation_map=navigation_map,
                                 vehicle_id=i) for i in range(number_of_agents)]

        
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

        result = {i: self.vehicles[i].move_vehicle(movement[0], movement[1]) for i ,movement in movements.items()}
        
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

    