import pandas as pd
from Environment.ShekelGroundTruth import Shekel
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

import numpy as np

class MetricsDataCreator:

	def __init__(self, metrics_names, algorithm_name, experiment_name, directory='./'):

		self.metrics_names = metrics_names
		self.algorithm_name = algorithm_name
		self.data = []
		self.directory = directory
		self.experiment_name = experiment_name
		self.base_df = None

	def register_step(self, run_num, step, metrics, algorithm_name = None):

		if algorithm_name is None:
			algorithm_name = self.algorithm_name

		""" Append the next step value of metrics """
		self.data.append([algorithm_name, run_num, step, *metrics])

	def register_experiment(self):

		df = pd.DataFrame(data = self.data, columns=['Algorithm', 'Run', 'Step', *self.metrics_names])

		if self.base_df is None:
			df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return df
		else:
			self.base_df = pd.concat((self.base_df, df), ignore_index = True)
			self.base_df.to_csv(self.directory + self.experiment_name + '.csv', sep = ',')
			return self.base_df

	def load_df(self, path):

		self.base_df = pd.read_csv(path, sep=',')


class BenchmarkEvaluator:

	def __init__(self, navigation_map, l = 0.75):

		self.navigation_map = navigation_map
		self.visitable_positions = np.column_stack(np.where(navigation_map == 1)).astype(float)
		self.gt = Shekel(map_lims = self.navigation_map.shape, visitable_positions = self.visitable_positions)
		self.gt.reset()
		self.visited_positions = None
		self.measured_values = None
		self.rmse = []

		kernel = Matern(length_scale=l)
		self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-7)

	def reset_values(self):

		self.visited_positions = None
		self.measured_values = None
		self.gt.reset()
		self.rmse = []

	def update_rmse(self, positions):

		if self.visited_positions is None:
			self.visited_positions = np.asarray(positions)
			self.measured_values = np.asarray([self.gt.evaluate(pos) for pos in positions]).reshape(-1,1)
		else:
			self.visited_positions = np.vstack((self.visited_positions, positions))
			self.measured_values = np.vstack((self.measured_values, np.asarray([self.gt.evaluate(pos) for pos in positions]).reshape(-1,1)))

		self.visited_positions, unq_indx = np.unique(self.visited_positions, return_index=True, axis=0)
		self.measured_values = self.measured_values[unq_indx]

		# Fit
		self.gpr.fit(self.visited_positions/self.navigation_map.shape, self.measured_values)
		# Predict
		predicted_values = self.gpr.predict(self.visitable_positions/self.navigation_map.shape)

		self.rmse.append(mean_squared_error(y_true = (self.gt.GroundTruth_field - self.gt.GT_mean)/(self.gt.GT_std + 1e-8), y_pred = predicted_values, squared = False))


		return self.rmse[-1], predicted_values






