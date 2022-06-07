from InformationGatheringEnvironments import QuadcopterAgent
import numpy as np
import matplotlib.pyplot as plt

agent_config = {'navigation_map': np.ones((100,100)),
	                'mask_size': (10,10),
	                'initial_position': np.array([50,50]),
	                'speed': 1,
	                'max_illegal_movements': 10,
	                'max_distance': 10000000,
	                'ground_truth_field': np.random.rand(100, 100),
	                'dt': 1}

agent = QuadcopterAgent(agent_config, 'pepe')

agent.reset()
agent.render()

def target_to_angledist(current, target):

	diff = target - current
	angle = np.arctan2(diff[1], diff[0])
	dist = np.linalg.norm(diff)

	return angle,dist

def onclick(event):

	global agent

	agent.stop_go_to()
	new_position = np.asarray([event.ydata, event.xdata])
	angle,dist = target_to_angledist(agent.position, new_position)
	agent.go_to_next_waypoint_relative(angle, dist)


cid = agent.fig.canvas.mpl_connect('button_press_event', onclick)

done = False

indx = 0

x,y = np.meshgrid(np.arange(10,80,10), np.arange(10,80,10))

wps = np.column_stack((y.flatten(),x.flatten()))

indx = 0

print("Waypoint ", wps[indx,:])
angle, dist = target_to_angledist(agent.position, wps[indx,:])
agent.go_to_next_waypoint_relative(angle, dist)

while True:


	_,done = agent.step()
	agent.render()
	plt.pause(0.01)


	if indx == len(wps) - 1:
		agent.reset()
		indx = 0

	if agent.wp_reached:

		indx += 1
		print("Waypoint ", wps[indx, :])
		angle, dist = target_to_angledist(agent.position, wps[indx,:])
		agent.go_to_next_waypoint_relative(angle, dist)