import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from deap import benchmarks
from mpl_toolkits.mplot3d import Axes3D

""" Gaussian Process Regressor """
gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.08, length_scale_bounds='fixed'), alpha=0, n_restarts_optimizer=5)

""" Benchmark parameters """
A = [[0.5,  0.5],
     [0.25, 0.25],
     [0.25, 0.75],
     [0.75, 0.25],
     [0.75, 0.75]]

C = [0.02, 0.02, 0.02, 0.005, 0.005]

def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]

def r_interest(val_mu):

    return np.max((0, 1 + val_mu))


""" Compute Ground Truth """
fig, ax = plt.subplots(1,3)
X = np.arange(0, 1, 0.02)
Y = np.arange(0, 1, 0.02)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
Z = (Z - Z.min())/(Z.max() - Z.min() + 1E-8)
x_locs = np.column_stack((X.flatten(),Y.flatten()))
x_meas = np.asarray([[0.0, 0.0]])
y_meas = np.asarray([Z[int(x_meas[-1, 0]), int(x_meas[-1,1])]])

""" Fit the data """
gp.fit(x_meas, y_meas)
mu, sigma = gp.predict(x_locs, return_cov=True)

""" Compute the entropy """
r = np.array(list(map(r_interest, mu.flatten())))
sigma_weighted = sigma + np.diag(r)
H1 = sigma_weighted.trace() / np.sqrt(sigma_weighted.shape[0])

print('Entropy: ', np.linalg.det(sigma_weighted))

""" Plot the data"""
d = ax[0].imshow(mu.reshape(Z.shape), cmap='jet', vmin=0, vmax=1)
ds = ax[1].imshow(sigma.diagonal().reshape(Z.shape), cmap='viridis')
d1, = ax[0].plot(x_meas[:,1]*50, x_meas[:,0]*50, 'r-x')
de = ax[2].imshow(Z, cmap='viridis')



def onclick(event):

    global x_meas
    global y_meas
    global gp
    global d
    global H

    """ Get the new measurement """
    x_meas = np.vstack((x_meas, [event.xdata/50, event.ydata/50]))
    y_meas = np.vstack((y_meas, [Z[int(x_meas[-1, 0]*50), int(x_meas[-1, 1]*50)]]))

    un_locs, indxs = np.unique(x_meas, return_index=True, axis=0)
    un_values = y_meas[indxs]

    """ Fit the GP """
    gp.fit(x_meas, y_meas)
    mu_, sigma_ = gp.predict(x_locs, return_cov=True)

    """ Compute the entropy """
    r = np.array(list(map(r_interest, mu_.flatten())))
    sigma_weighted = sigma_ + np.diag(r)

    mu_ = mu_.reshape(Z.shape)
    H_new = sigma_weighted.trace() / np.sqrt(sigma_weighted.shape[0])
    print('Entropy: ', H1/H_new)


    """ Plot the results """
    d.set_data(mu_)
    d1.set_xdata(x_meas[:,0]*50)
    d1.set_ydata(x_meas[:,1]*50)
    ds.set_data(sigma_.diagonal().reshape(Z.shape))
    de.set_data(Z)
    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
