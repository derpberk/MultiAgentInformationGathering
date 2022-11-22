""" This script tend to find the best hyperparameters for the Gaussian process model. """
import sys
sys.path.append('.')
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from Environment.GroundTruths.FireFront import WildfireSimulator
import multiprocessing

from Environment.groundtruth import GroundTruth

navigation_map = np.genfromtxt('Environment/SquaredMap.txt')
visitable_positions = np.column_stack(np.where(navigation_map == 1))

def optimize_environment(i):

	print("PROCES NUM ", i)

	theta = []
	gp = GaussianProcessRegressor(kernel=RBF(length_scale=5.0, length_scale_bounds=(0.01, 10000)), alpha=0.001, n_restarts_optimizer=50)
	visitable_positions = np.column_stack(np.where(navigation_map == 1))
	sim_config = WildfireSimulator.sim_config_template
	sim_config['navigation_map'] = np.genfromtxt('Environment/SquaredMap.txt')
	ground_truth = WildfireSimulator(sim_config)

	for i in range(10):

		ground_truth.reset()
		random_positions_index = np.arange(0,len(visitable_positions),step=4)
		x_meas = visitable_positions[random_positions_index]
		y_meas = ground_truth.read()[x_meas[:, 0], x_meas[:, 1]]
		gp.fit(x_meas, y_meas)
		theta.append(np.copy(gp.kernel_.length_scale))

	return theta



if __name__ == '__main__':

	pool = multiprocessing.Pool()

	"""
	with multiprocessing.Pool() as pool:
		T = np.asarray(pool.map(optimize_environment, range(100))).flatten()

	pool.close()
	"""

	optimize_environment(1)

	with plt.style.context('seaborn-darkgrid'):
		plt.hist(T, density=True)
		plt.xlabel('lengthscale of RBF $(l)$')
		plt.ylabel('Density')
		np.savetxt('./lengthscale_histogram_wildires.txt', T)
		plt.show()

