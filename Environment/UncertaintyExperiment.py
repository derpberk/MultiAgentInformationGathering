import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from deap import benchmarks
from mpl_toolkits.mplot3d import Axes3D

""" Gaussian Process Regressor """
gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.08, length_scale_bounds='fixed'), alpha=0.01, n_restarts_optimizer=5)
A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
C = [0.002, 0.5, 0.5, 0.5, 0.5]

def shekel_arg0(sol):
    return benchmarks.shekel(sol, A, C)[0]

""" Compute Ground Truth """
fig, ax = plt.subplots(1,3)
X = np.arange(0, 1, 0.02)
Y = np.arange(0, 1, 0.02)
X, Y = np.meshgrid(X, Y)
Z = np.fromiter(map(shekel_arg0, zip(X.flat,Y.flat)), dtype=np.float, count=X.shape[0]*X.shape[1]).reshape(X.shape)*0.02
x_locs = np.column_stack((X.flatten(),Y.flatten()))
x_meas = np.asarray([[0.0, 0.0]])
y_meas = np.asarray([Z[int(x_meas[-1, 0]), int(x_meas[-1,1])]])

""" Fit the data """
gp.fit(x_meas, y_meas)
mu, sigma = gp.predict(x_locs, return_cov=True)
print(gp.kernel_)
mu = mu.reshape(Z.shape)
H = sigma.trace()/np.sqrt(sigma.shape[0])
print('Entropy: ', H)

""" Plot the data"""
d = ax[0].imshow(Z, cmap='jet')
ds = ax[1].imshow(sigma.diagonal().reshape(Z.shape), cmap='viridis', vmin=1.2, vmax=3)
d1, = ax[0].plot(x_meas[:,1]*50, x_meas[:,0]*50, 'r-x')
de = ax[2].imshow(sigma, cmap='viridis')

def r_interest(val_mu):

    return np.min((5, (1.5-1) * val_mu +1))


def onclick(event):

    global x_meas
    global y_meas
    global gp
    global d
    global H

    x_meas = np.vstack((x_meas, [event.xdata/50, event.ydata/50]))
    y_meas = np.vstack((y_meas, [Z[int(x_meas[-1, 0]*50), int(x_meas[-1, 1]*50)]]))
    gp.fit(x_meas, y_meas)
    print(gp.kernel_)
    mu_, sigma = gp.predict(x_locs, return_cov=True)
    r = np.array(list(map(r_interest, mu_.flatten())))
    sigma = sigma + np.diag(r)
    sigma = sigma
    mu_ = mu_.reshape(Z.shape)
    H_new = sigma.trace()/np.sqrt(sigma.shape[0])
    print('Entropy: ',H-H_new)
    H = H_new
    d.set_data(mu_)
    d1.set_xdata(x_meas[:,0]*50)
    d1.set_ydata(x_meas[:,1]*50)
    ds.set_data(sigma.diagonal().reshape(Z.shape))
    de.set_data(sigma)
    fig.canvas.draw()
    print(sigma.diagonal())


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
