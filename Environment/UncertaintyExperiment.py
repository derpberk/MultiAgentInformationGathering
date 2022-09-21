import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from Environment.OilSpillEnvironment import OilSpill
from Environment.ShekelGroundTruth import Shekel
from Environment.FireFront import WildfireSimulator
from Environment.GasSource.GasSourceGroundTruth import GasSourceGT
from sklearn.metrics import mean_squared_error

from mpl_toolkits.mplot3d import Axes3D


max_distance = 3.0

temporal = True
""" Gaussian Process Regressor """
if temporal:
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=(2.5, 2.5, 60), length_scale_bounds=[(3.5, 3.5), (3.5, 3.5), (60, 60)]), optimizer=None, alpha=0.001)

else:
    gp = GaussianProcessRegressor(kernel=Matern(length_scale=3.5, length_scale_bounds=(0.1, 10.)), alpha=0.001, n_restarts_optimizer=20, optimizer=None,)


config = WildfireSimulator.sim_config_template
config['navigation_map'] = np.ones((50, 50))
config['init_time'] = 0

gt = WildfireSimulator(config)
gt.reset(True)
gt.step()

""" Compute Ground Truth """
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(nrows=2, ncols=3)
ax = []
ax.append(fig.add_subplot(gs[0, 0]))
ax.append(fig.add_subplot(gs[0, 1]))
ax.append(fig.add_subplot(gs[0, 2]))
ax.append(fig.add_subplot(gs[1, :]))

x_locs = np.column_stack(np.where(gt.sim_config_template['navigation_map'] == 1))

x_meas = np.asarray([[0.0, 0.0]])

t_sample = np.zeros((1, 1))

y_meas = np.array([gt.read(position=x_meas[-1].astype(int))])


position = np.array([0.1, 0.1])

""" Fit the data """
if temporal:
    gp.fit(np.hstack((x_meas, t_sample)), y_meas)
    mu, sigma = gp.predict(np.hstack((x_locs, np.zeros((x_locs.shape[0],1)))), return_std=True)
else:
    gp.fit(x_meas, y_meas)
    mu, sigma = gp.predict(x_locs, return_std=True)

error = []
error.append(mean_squared_error(gt.ground_truth_field.flatten(), mu))

""" Plot the data"""
d = ax[0].imshow(gt.read(), cmap='jet', vmax=1.0, vmin=0.0)
ds = ax[1].imshow(sigma.reshape(gt.ground_truth_field.shape), cmap='viridis')
d1, = ax[0].plot(x_meas[:,1]*50, x_meas[:,0]*50, 'r-x', markersize=2, linewidth=0.1)
de = ax[2].imshow(gt.read(), cmap='jet', vmax=1.0, vmin=0.0)
derr, = ax[3].plot(error, 'x-')

ax[3].set_xlim((0, 100))
ax[3].set_ylim((0, error[0]))
ax[3].grid()
ax[3].set_ylabel('Error RMSE')
axt = ax[3].twinx()
axt.set_ylabel('Uncertainty')


H = sigma.sum()
R = []
Re = []
He = []
Ht = [H]

H0 = sigma.sum()

axt.set_ylim((0, sigma.mean()))
du, = axt.plot(sigma.mean(), 'r-o')
sigma_ant = sigma.copy()


def get_measurement(gt_field, point, size, step):

    x = np.arange(point[0] - size, point[0] + size).astype(int)
    y = np.arange(point[1] - size, point[1] + size).astype(int)

    positions = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    values = gt_field[positions[:, 0], positions[:, 1]]

    return positions, values

def onclick(event):

    global x_meas
    global y_meas
    global gp
    global d
    global H, R, Re, He, sigma_ant, t_sample, position

    """ Get the new measurement """
    new_point = np.array([event.ydata, event.xdata])

    new_direction = new_point - position
    new_normalized_direction = new_direction/np.linalg.norm(new_direction)
    add_point = position + new_normalized_direction*max_distance

    position = add_point

    # new_measurement = Z[(add_point[0]*Z.shape[0]).astype(int), (add_point[1]*Z.shape[1]).astype(int)][np.newaxis]

    new_locations, new_measurement = get_measurement(gt.read(), add_point, size=2, step=6)

    x_meas = np.vstack((x_meas, new_locations))
    y_meas = np.concatenate((y_meas, new_measurement))


    """ Fit the GP """
    if temporal:

        t_sample += 1
        t_sample = np.vstack((t_sample, np.zeros_like(new_measurement).reshape(-1,1)))
        forgot_indexes = np.where(t_sample.flatten() < 50)

        x_meas = x_meas[forgot_indexes]
        y_meas = y_meas[forgot_indexes]
        t_sample = t_sample[forgot_indexes]

        features, unique_indx = np.unique(np.hstack((x_meas, t_sample)), axis=0, return_index=True)
        y_meas = y_meas[unique_indx]
        x_meas = x_meas[unique_indx]
        t_sample = t_sample[unique_indx]
        gp.fit(features, y_meas)
        mu_, sigma_ = gp.predict(np.hstack((x_locs, np.zeros((x_locs.shape[0],1)))), return_std=True)

        gt.step()

    else:

        x_meas, unique_indx = np.unique(x_meas, axis=0, return_index=True)
        y_meas = y_meas[unique_indx]
        gp.fit(x_meas, y_meas)
        mu_, sigma_ = gp.predict(x_locs, return_std=True)

    sigma_ant = sigma_.copy()

    error.append(mean_squared_error(gt.ground_truth_field.flatten(), mu_))


    """ Compute the uncertainty decrement """
    uncertainty_decrement = 100*(H - sigma_.sum())/H0
    He.append(uncertainty_decrement)

    print('Entropy: ', H - sigma_.sum())
    H = sigma_.sum()
    Ht.append(sigma_.mean())

    """ Compute the Regret """

    regret = y_meas[-1] - y_meas[:-1].max()
    Re.append(regret * uncertainty_decrement)

    print('Regret: ', regret)
    print('Lengthscale', gp.kernel_.length_scale)

    """ Compute the reward """
    reward = (1+y_meas[-1]) * uncertainty_decrement
    R.append(reward)
    print('Reward', reward)

    """ Plot the results """
    d.set_data(mu_.reshape(gt.ground_truth_field.shape))
    d1.set_xdata(x_meas[:, 1])
    d1.set_ydata(x_meas[:, 0])
    ds.set_data(sigma_.reshape(gt.ground_truth_field.shape))
    de.set_data(gt.ground_truth_field)

    derr.set_xdata(np.arange(0,len(error)))
    derr.set_ydata(error)

    du.set_xdata(np.arange(0, len(Ht)))
    du.set_ydata(Ht)

    fig.canvas.draw()
    fig.canvas.flush_events()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.ioff()
plt.ion()
plt.show(block=True)

