import numpy as np
import scipy.interpolate as interpolate


def is_valid(point, navigation_map):

	low = np.floor(point).astype(int)
	high = np.ceil(point).astype(int)

	return navigation_map[low[0],low[1]] == 1 and \
	       navigation_map[low[0],high[1]] and \
	       navigation_map[high[0],low[1]] and \
	       navigation_map[high[0],high[1]]

def create_valid_path(navigation_map, n_points, starting_point=None, meas_distance=1):
	""" Create a valid path given a navigation map """

	N = 500
	# Find visitable points #
	visitable_points = np.column_stack(np.where(navigation_map == 1))

	# Randomly generate paths until there is a feasible path #
	valid = False

	waypoints = None
	interp_path = None
	while not valid:
		# Choose n_points unique random points that are visitable #
		waypoints_indexes = np.random.choice(np.arange(len(visitable_points)), n_points, replace=False)
		waypoints = visitable_points[waypoints_indexes]

		if starting_point is not None:
			waypoints = np.vstack(([starting_point], waypoints))

		# Interpolate #
		tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1]], s=0.0, k=1)
		interp_path = np.column_stack(interpolate.splev(np.linspace(0, 1, N), tck))

		interp_path = np.clip(interp_path, 0, np.asarray(nav_map.shape)-1)

		valid = all([is_valid(point, navigation_map) for point in interp_path])

	# Slice waypoints to have points that are at a distance of meas_distance from the previous ones #
	path_diff = np.diff(interp_path, axis=0)  # Diff vectors
	distance_path = np.linalg.norm(path_diff, axis=1)
	length = np.sum(distance_path)

	subpoints_path = interp_path[::int(meas_distance*len(interp_path)/length)]

	return subpoints_path, waypoints


if __name__ == '__main__':
	
	import matplotlib.pyplot as plt
	
	plt.ion()
	
	nav_map = np.genfromtxt('../../Environment/wesslinger_map.txt')
	
	ways, path = create_valid_path(nav_map, 20, starting_point=[30, 13], meas_distance=3)
	
	plt.imshow(nav_map, cmap='gray')
	plt.plot(ways[:, 1], ways[:, 0], 'go-')
	plt.plot(path[:, 1], path[:, 0], 'rx')

	plt.show(block=True)










