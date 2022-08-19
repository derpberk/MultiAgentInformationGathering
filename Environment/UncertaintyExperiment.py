import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from deap import benchmarks
from mpl_toolkits.mplot3d import Axes3D


max_distance = 0.1
temporal = False
""" Gaussian Process Regressor """
if temporal:
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=(0.1, 0.1, 20), length_scale_bounds=[(0.1,0.1),(0.1,0.1),(20,20)]), alpha=0.01)

else:
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.5, length_scale_bounds=(0.1, 0.1)), alpha=0.00001, n_restarts_optimizer=20)

""" Benchmark parameters """
A = [[0.5,  0.5],
     [0.25, 0.25],
     [0.25, 0.75],
     [0.75, 0.25],
     [0.75, 0.75]]

C = [0.02, 0.02, 0.02, 0.005, 0.005]

def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]


""" Compute Ground Truth """
fig, ax = plt.subplots(1,3)
X = np.arange(0, 1, 0.02)
Y = np.arange(0, 1, 0.02)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
Z = np.asarray((Z - Z.min())/(Z.max() - Z.min() + 1E-8))
x_locs = np.column_stack((X.flatten(),Y.flatten()))
x_meas = np.asarray([[0.0, 0.0],
                     [0.1, 0.1]])

t_sample = np.zeros((2, 1))

y_meas = Z[(x_meas[:,0]*50).astype(int), (x_meas[:,1]*50).astype(int)]



""" Fit the data """
if temporal:
    gp.fit(np.hstack((x_meas, t_sample)), y_meas)
    mu, sigma = gp.predict(np.hstack((x_locs, np.zeros((x_locs.shape[0],1)))), return_std=True)
else:
    gp.fit(x_meas, y_meas)
    mu, sigma = gp.predict(x_locs, return_std=True)

""" Plot the data"""
d = ax[0].imshow(mu.reshape(Z.shape), cmap='jet', vmax=1.0, vmin=0.0)
ds = ax[1].imshow(sigma.reshape(Z.shape), cmap='viridis')
d1, = ax[0].plot(x_meas[:,1]*50, x_meas[:,0]*50, 'r-x')
de = ax[2].imshow(Z.T, cmap='jet', vmax=1.0, vmin=0.0)

H = sigma.sum()
R = []
Re = []
He = []

H0 = sigma.sum()

sigma_ant = sigma.copy()

def onclick(event):

    global x_meas
    global y_meas
    global gp
    global d
    global H, R, Re, He, sigma_ant, t_sample

    """ Get the new measurement """
    new_point = np.array([event.xdata/50, event.ydata/50])

    new_direction = new_point - x_meas[-1]
    new_normalized_direction = new_direction/np.linalg.norm(new_direction)
    add_point = x_meas[-1] + new_normalized_direction*max_distance

    new_measurement = Z[(add_point[0]*Z.shape[0]).astype(int), (add_point[1]*Z.shape[1]).astype(int)][np.newaxis]

    x_meas = np.vstack((x_meas, add_point.copy()))
    y_meas = np.concatenate((y_meas, new_measurement))

    t_sample += 1
    t_sample = np.vstack((t_sample, np.array([0])))


    """ Fit the GP """
    if temporal:
        gp.fit(np.hstack((x_meas, t_sample)), y_meas)
        mu_, sigma_ = gp.predict(np.hstack((x_locs, np.zeros((x_locs.shape[0],1)))), return_std=True)
    else:
        gp.fit(x_meas, y_meas)
        mu_, sigma_ = gp.predict(x_locs, return_std=True)
    sigma_ant = sigma_.copy()


    """ Compute the uncertainty decrement """
    uncertainty_decrement = 100*(H - sigma_.sum())/H0
    He.append(uncertainty_decrement)

    print('Entropy: ', H - sigma_.sum())
    H = sigma_.sum()

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
    d.set_data(mu_.reshape(50, 50))
    d1.set_xdata(x_meas[:, 0]*50)
    d1.set_ydata(x_meas[:, 1]*50)
    ds.set_data(sigma_.reshape(Z.shape))
    de.set_data(Z.T)
    fig.canvas.draw()
    fig.canvas.flush_events()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.ioff()
plt.ion()
plt.show(block=True)


with plt.style.context('seaborn-whitegrid'):

    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(R, 'b-o'); axs[0].set_ylabel('Reward')
    axs[1].plot(y_meas[2:].flatten(), 'r-o'); axs[1].set_ylabel('Measurement value')
    axs[2].plot(Re, 'g-o'); axs[2].set_ylabel('Regret')
    axs[3].plot(He, 'k-o'); axs[3].set_ylabel('Uncertainty')
    plt.show(block=True)

    print("TOTAL REWARD: ", np.sum(R))
