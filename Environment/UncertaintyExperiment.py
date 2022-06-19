import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from deap import benchmarks
from mpl_toolkits.mplot3d import Axes3D

""" Gaussian Process Regressor """
gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.08, length_scale_bounds=(0.01, 0.5)), alpha=0.01, optimizer=None)

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
y_meas = Z[(x_meas[:,0]*50).astype(int), (x_meas[:,1]*50).astype(int)]


""" Fit the data """
gp.fit(x_meas, y_meas)
mu, sigma = gp.predict(x_locs, return_cov=True)

""" Compute the entropy """
H_ = sigma.trace()
H1 = H_
y_max = np.max(y_meas)

print('Entropy: ', np.linalg.det(sigma))

""" Plot the data"""
d = ax[0].imshow(mu.reshape(Z.shape), cmap='jet', vmin=0, vmax=1)
ds = ax[1].imshow(np.sqrt(sigma.diagonal()).reshape(Z.shape), cmap='viridis')
d1, = ax[0].plot(x_meas[:,1]*50, x_meas[:,0]*50, 'r-x')
de = ax[2].imshow(Z.T, cmap='viridis')

def onclick(event):

    global x_meas
    global y_meas
    global gp
    global d
    global H, H_

    """ Get the new measurement """
    x_meas = np.vstack((x_meas, [event.xdata/50, event.ydata/50]))
    y_meas = np.concatenate((y_meas, Z[(x_meas[-1,0]*50).astype(int), (x_meas[-1,1]*50).astype(int)][np.newaxis]))

    un_locs, indxs = np.unique(x_meas, return_index=True, axis=0)
    un_values = y_meas[indxs]

    #new_lengthscale = np.mean(H_)/H1 * 0.08
    #gp.set_params(kernel__length_scale=new_lengthscale)

    """ Fit the GP """
    gp.fit(x_meas, y_meas)
    mu_, sigma_ = gp.predict(x_locs, return_cov=True)


    mu_ = mu_.reshape(Z.shape)
    H_new = sigma_.trace()
    print('Entropy: ', H_ - H_new)

    regret = 1 + (y_meas[-1] - y_meas.max())

    print('Regret: ', regret)

    print("Reward: ", (H_ - H_new)/49*regret)

    H_ = H_new

    """ Plot the results """
    d.set_data(mu_)
    d1.set_xdata(x_meas[:,0]*50)
    d1.set_ydata(x_meas[:,1]*50)
    ds.set_data(np.sqrt(sigma_.diagonal()).reshape(Z.shape))
    de.set_data(Z.T)
    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
