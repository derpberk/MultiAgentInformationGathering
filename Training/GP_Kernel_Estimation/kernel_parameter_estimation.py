""" This script tend to find the best hyperparameters for the Gaussian process model. """
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from Environment.ShekelGroundTruth import Shekel

navigation_map = np.genfromtxt('../../Environment/wesslinger_map.txt')
N = 4

same_evaluation_scenario = False


ground_truth = Shekel(1-navigation_map, 1, max_number_of_peaks=5, is_bounded=True, seed=0)

T = 20  # Number of iterations
N = 200 # Number of points for the GP
theta = [] # List of theta values
N_envs = 100

gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.08, length_scale_bounds=(0.01, 10000)), alpha=0.01, n_restarts_optimizer=50)

visitable_positions = np.column_stack(np.where(navigation_map == 1))

for i in range(N_envs):

	ground_truth.reset()

	for t in range(T):

		random_positions_index = np.random.choice(np.arange(0, len(visitable_positions)), size=N, replace=False)

		x_meas = visitable_positions[random_positions_index]
		y_meas = ground_truth.read()[x_meas[:, 0], x_meas[:, 1]]

		gp.fit(x_meas,y_meas)

		theta.append(gp.kernel_.length_scale)

		print("Iteration: ", t, "Theta: ", gp.kernel_.length_scale)


with plt.style.context('seaborn-darkgrid'):
	plt.hist(theta, density=True)
	plt.xlabel('lengthscale of RBF $(l)$')
	plt.ylabel('Density')
	np.savetxt('./lengthscale_histogram_shekel.txt', theta)
	plt.show()
