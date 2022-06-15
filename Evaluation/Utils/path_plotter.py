import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import interpolate



def plot_trajectory(env_map, trajectories):

	plt.style.use('grayscale')

	plt.ion()

	styles = ['solid', 'dashed', 'dashdot', 'dotted']

	env_map = ndimage.binary_dilation(env_map, [[False, True, False], [True, True, True], [False, True, False]])


	n_trajs = int(trajectories.shape[1]/2)

	fig, ax = plt.subplots(1, 1)

	ax.imshow(env_map, cmap='gray')
	ax.get_yaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)

	for idx, a in enumerate(range(0,n_trajs*2,2)):
		xp = trajectories[:, a + 1]
		yp = trajectories[:, a]

		okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)

		xp = xp[okay]
		yp = yp[okay]

		tck, u = interpolate.splprep([xp, yp], s=0.0)
		x_i, y_i = interpolate.splev(np.linspace(0, 1, 300), tck)

		ax.plot(x_i, y_i, ls=styles[idx], lw=1.5, label = f'Agent {idx+1}')


	plt.legend()
	plt.show()

