import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

plt.style.use('bmh')

# objective function
def objective(x_):
	return -(1.4 - 3.0 * x_) * np.sin(18.0 * x_)

# GP #
kernel = ConstantKernel(constant_value=1.0, constant_value_bounds='fixed') * RBF(length_scale=0.1, length_scale_bounds='fixed')
GP = GaussianProcessRegressor(kernel=kernel, alpha=0.1, optimizer=None)

# Set the true value #
X = np.linspace(0,1.2,200)
Y_true = objective(X)

fig, axs = plt.subplots(2,1)

axs[0].plot(X, Y_true, 'b--', label = 'True f(x)')

# Fit the GP without alpha

x = np.linspace(0,1.2,50)
y = objective(x)
GP.fit(x.reshape(-1,1),y)
Y,s = GP.predict(X.reshape(-1,1),return_std=True)
axs[0].fill_between(X,Y+s, Y-s, color='red', alpha=0.15)
axs[0].plot(x,y, 'kx', label='Sampled points')
axs[0].plot(X,Y, color='red', label='Inferred f(x)')
axs[1].plot(X,s, 'r')


# Add noise #


# Add some intensification #
N = 20
x_new = np.concatenate((x, (np.random.rand(N)-0.5)*0.15 + 0.08))
y_new = objective(x_new)
abs_y = np.abs(y_new)
alpha = np.clip(1-abs_y/np.max(abs_y), 0.0001, 10)
GP.set_params(alpha=alpha)

GP.fit(x_new.reshape(-1,1), y_new)
Y,s = GP.predict(X.reshape(-1,1), return_std=True)
axs[0].fill_between(X,Y+s, Y-s, color='green', alpha=0.15)
axs[0].plot(X,Y, color='green', label='Inferred f(x) with alpha')
axs[0].plot(x_new[-N:],y_new[-N:], 'm^', label='Additional points')
axs[1].plot(X,s, 'g')
axs[0].legend()




plt.show()
