import sys
sys.path.append('.')

""" These functions generate a environment according to the design cases. """
from Environment.StaticIGEnvironments import StaticIGEnv
from Environment.GroundTruths.ShekelGroundTruth import Shekel
from Environment.GroundTruths.FireFront import WildfireSimulator

import numpy as np

def generate_WQ_env(reward_type : str) -> StaticIGEnv:

    """ Generates an environment for the Water Quality Parameter situation
        using the Wesslinger Lake. """
    
    N = 4 
    nav_map = np.genfromtxt('Environment/NavigationMaps/ypacarai_map.txt')
    gt_config = Shekel.sim_config_template
    gt_config["navigation_map"] = nav_map
    gt = Shekel(gt_config)

    env = StaticIGEnv(number_of_vehicles = N,
                         navigation_map = nav_map,
                         max_travel_distance = 45,
                         initial_vehicle_positions = np.array([[20,20],[25,25],[30,30],[10,10]]),
                         movement_type = "DISCRETE",
                        movement_limits = (3,8),
                        benchmark = gt,
                        max_collisions = 10,
                        # Model parameters #
                        kernel_type = "Matern",
                        lengthscale = 12.5,
                        # Reward style #
                        reward_type = reward_type,
                        # Default parameters #
                        number_of_movements = 8,
                 )
    
    return env

def generate_FF_env(reward_type : str) -> StaticIGEnv:

    """ Generates an environment for the FireFront surveillance """
    
    N = 2 
    nav_map = np.genfromtxt('Environment/NavigationMaps/squared_map.txt')
    nav_map = np.ones((30,30))
    nav_map[0,:] = 0
    nav_map[-1,:] = 0
    nav_map[:,0] = 0
    nav_map[:,-1] = 0

    gt_config = WildfireSimulator.sim_config_template
    gt_config["navigation_map"] = nav_map
    gt = WildfireSimulator(gt_config)

    env = StaticIGEnv(number_of_vehicles = N,
                         navigation_map = nav_map,
                         max_travel_distance = 50,
                         initial_vehicle_positions = np.array([[5,5],[25,25],[5,25],[25,5]]),
                         movement_type = "DISCRETE",
                         movement_limits = (1,8),
                         benchmark = gt,
                         max_collisions = 10,
                         # Model parameters #
                         kernel_type = "Matern",
                         lengthscale = 1.5,
                         # Reward style #
                         reward_type = reward_type,
                         # Default parameters #
                         number_of_movements = 8,
                         agent_size=1,
                        
                 )
    
    return env

if __name__ == '__main__':

    from OtherAgents.NRRA import WanderingAgent
    from OtherAgents.LawnMower import LawnMowerAgent

    environment = generate_FF_env('Change')

    environment.reset()
    environment.render()

    agents = [LawnMowerAgent(world=environment.navigation_map, number_of_actions=8, movement_length = 2, forward_direction=2) for i in range(environment.number_of_agents)]
    agents = [WanderingAgent(world=environment.navigation_map, number_of_actions=8, movement_length = 2, consecutive_movements=10) for i in range(environment.number_of_agents)]
    agents_ids = [i for i in range(environment.number_of_agents)]

    for i in range(200):


        action = {j: agents[j].move(environment.fleet.fleet_position[j].astype(int)) for j in agents_ids}

        _,r,_,_ = environment.step(action)
        environment.render()

        print(r)