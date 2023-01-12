import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from Environment.EnvironmentCreation import generate_FF_env, generate_WQ_env

# EXECUTION CONDITIONS #
RUNS = 50
RENDER = True
RENDER_PATH = False

# Generate environment #
env = generate_WQ_env()

for run in range(RUNS):
     
    # Test a random agent #
    env.reset()
    done = {i: False for i in range(env.number_of_agents)}
    actions = {i:env.get_safe_action(i) for i in range(env.number_of_agents)}

    while not any(list(done.values())):

        env.render()

        for i in range(env.number_of_agents):
            if not env.get_action_mask(i)[actions[i]]:
                actions[i] = env.get_safe_action(i)
                
        s,r,done,i = env.step(actions)

        if RENDER:
            env.render()

    if RENDER_PATH:
        plt.imshow(env.navigation_map)
        for vehicle in env.fleet.vehicles:
            traj = np.asarray(vehicle.wps)
            plt.plot(traj[:,1], traj[:,0], '.-')
        plt.show(block=True)









    



