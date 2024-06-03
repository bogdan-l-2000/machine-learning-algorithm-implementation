import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random

# This is the Gaussian function.
# I am using this function from the online "Mathematics for Machine Learning: Multivariate Calculus" course offered by Imperial College London on Coursera
def GaussianFunction (x,mu,sig) :
    return np.exp(-(x-mu)**2/(2*sig**2)) / np.sqrt(2*np.pi) / sig

# Next up, the derivative with respect to μ.
# Note that the derivative calls the original function due to the derivative of e^x being equal to e^x
def GaussianDerivativeMu (x,mu,sig) :
    return GaussianFunction(x, mu, sig) * ((x - mu) / (sig ** 2))

# Finally in this cell, the derivative with respect to σ.
# Same idea as for mu
def GaussianDerivativeSig (x,mu,sig) :
    return GaussianFunction(x, mu, sig) * ((-1/sig) + (((x-mu)**2) / (sig ** 3)))


# Generate random (noisy) data and plot it, for visualization

# np.linspace:
# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# Use random values here
xdata = np.linspace(0, 100, random.randint(80, 120))
y = GaussianFunction(xdata, 50, 20)
np.random.seed(2421)
y_noise = 0.005 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

# Plot optimized function using scipy
# curve_fit: uses non-linear least squares to fit a function to data
# Returns popt, the optimal values for the parameters to minimize the sum of squared residuals
# Returns pcov, the estimated covariance of popt.
popt, pcov = scipy.optimize.curve_fit(GaussianFunction, xdata, ydata)
plt.plot(xdata, GaussianFunction(xdata, *popt), 'r-',
         label='scipy_fit')


# Use our own optimizer functions using multivariate calculus
# Idea: the steepest descent will move around in parameter space proportional to the negative of te Jacobian.
def steepest_step (x, y, mu, sig, aggression):
    # We find the gradient of chi-squared and want to set it to a minimum
    # Note that chi-squared is the sum of the squares of the residuals r
    J = np.array([
        -2*(y - GaussianFunction(x,mu,sig)) @ GaussianDerivativeMu(x,mu,sig),
        -2*(y - GaussianFunction(x,mu,sig)) @ GaussianDerivativeSig(x,mu,sig)
    ])
    step = -J * aggression
    return step

# Next we'll assign trial values for these.
mu = 30 ; sig = 3
# We'll keep a track of these so we can plot their evolution.
p = np.array([[mu, sig]])

for i in range(2000) :
    dmu, dsig = steepest_step(xdata, ydata, mu, sig, 45)
    mu += dmu
    sig += dsig
    p = np.append(p, [[mu,sig]], axis=0)

print(p)

plt.plot(xdata, GaussianFunction(xdata, mu, sig), 'g-',
         label='steepest_step')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# plt.hist(f, [mu, sig])
