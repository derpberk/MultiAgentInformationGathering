import numpy as np
import scipy.sparse
from scipy.sparse import vstack, diags
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from Environment.groundtruth import GroundTruth
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator

import time

class GasSourceGT(GroundTruth):

	sim_config_template = {
		"seed": 0,
	}

	def __init__(self, init_config):
		super().__init__(init_config)

		# load operators
		self.fig = None
		self.ax = None
		self.d = None

		self.L = scipy.sparse.load_npz('./GasSource/laplace.npz')
		self.Gx = scipy.sparse.load_npz('./GasSource/grad_x.npz')
		self.Gy = scipy.sparse.load_npz('./GasSource/grad_y.npz')

		# load mesh
		self.verts = np.load('./GasSource/vertices.npy')
		self.dim = self.verts.shape[0]

		# find border
		eps = 0.5 * 1. / self.dim
		self.b = 1. * np.logical_or(np.logical_or(self.verts[:, 0] < eps, self.verts[:, 0] > 1 - eps), np.logical_or(self.verts[:, 1] < eps, self.verts[:, 1] > 1 - eps))

		# set boundary condition, i.e. =0
		self.R = diags(self.b)
		self.r = np.zeros((self.dim, 1))

		self.dt = 1000.0

		self.f_ = np.zeros((self.dim, 1))
		self.number_of_sources = None
		self.source = None
		self.wind_x, self.wind_y = None, None
		self.kappa = 0.005

		self.triObj = Triangulation(self.verts[:, 0], self.verts[:, 1])

	def reset(self, random_gt: bool = False):

		if random_gt:
			# parameter

			self.number_of_sources = np.random.randint(1, 3)
			self.source = np.random.rand(self.number_of_sources, 2)

			self.wind_x = 2 * (np.random.rand() - 0.5)
			self.wind_y = 2 * (np.random.rand() - 0.5)
			self.kappa = np.random.normal(0.02, 0.0005)

		else:
			self.number_of_sources = np.random.RandomState(self.seed).randint(1, 3)
			self.source = np.random.RandomState(self.seed).rand(self.number_of_sources, 2)

			self.wind_x = 2 * (np.random.RandomState(self.seed).rand() - 0.5)
			self.wind_y = 2 * (np.random.RandomState(self.seed).rand() - 0.5)

			self.kappa = np.random.RandomState(self.seed).normal(0.05, 0.005)

		self.step()

	def step(self):

		# setup right hand side
		s = np.zeros((self.dim, 1))

		for j in range(len(self.source)):
			idx = np.argmin(np.linalg.norm(self.verts - self.source[j], axis=1))
			s[idx, 0] = 1.

		y = s + (1.0 / self.dt * self.f_).reshape(-1, 1)

		# PDE
		O = 1.0 / self.dt * np.eye(self.dim) + self.kappa * self.L + self.wind_x * self.Gx + self.wind_y * self.Gy

		# linear system of equations
		A = vstack([O[np.logical_not(self.b)], self.R])
		c = vstack([y[np.logical_not(self.b)], self.r])

		# solve
		f = lsqr(A, c.toarray())[0]
		self.f_ = f

		self.wind_x += 2 * (np.random.rand() - 0.5) * 0.01
		self.wind_y += 2 * (np.random.rand() - 0.5) * 0.01

		# linear interpolation
		fz = LinearTriInterpolator(self.triObj, f)
		X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50),)
		Z = fz(X, Y)

		self.ground_truth_field = Z / (1e-6 + Z.max())

	def render(self):

		if self.fig is None:
			self.fig, self.ax = plt.subplots(1,1)
			self.d = plt.imshow(self.ground_truth_field, vmin=0.0, vmax=1.0, cmap='jet')
		else:
			self.d.set_data(self.ground_truth_field)

		plt.draw()
		plt.pause(0.01)

	def update_to_time(self, t):

		for _ in range(t):
			self.step(True)

		return self.ground_truth_field

	def read(self, position=None):

		if position is None:
			return self.ground_truth_field
		else:
			return self.ground_truth_field[int(position[0]), int(position[1])]




if __name__ == '__main__':

	gt = GasSourceGT(GasSourceGT.sim_config_template)
	gt.reset(True)
	for t in range(100):

		gt.step()
		gt.render()