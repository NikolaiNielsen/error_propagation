import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt


def error_propagation(expression, symbols, mu, sigma):
	"""
	General error propagation for one formula, given n variables.
	inputs:
	- expression: string of the expression, eg "m*v**2 / 2"
	- symbols: list of strings denoting the variables, eg ["m", "v"]
	- mu: list of most probable values for the variables, eg [1, 2]
	- sigma: covariance matrix for the variables, or list of variances
			 if there are no correlations
	It is important that the variables "symbols", "mu", and "sigma" 
	observe the same ordering. Ie, mu[0] is the most probable value of 
	symbols[0], and sigma[0,0] is the variance of symbols[0].

	Outputs:
	- std: standard deviation of the given expression.

	Taken from R. J. Barlows Statistics, chapter 4.3.4.
	"""

	# Check whether sigma is a matrix or list:
	if isinstance(sigma, list) or len(sigma.shape) == 1:
		# It is a list, convert to diagonal matrix
		sigma = np.diag(sigma)

	# Differentiate the expression wrt the symbols
	functs = [sp.diff(expression, sym) for sym in symbols]
	# evaluate the derivatives at the point mu, and convert to a vector
	vals = [float(f.subs(zip(symbols, mu))) for f in functs]
	vals = np.array(vals).reshape((len(vals), 1))

	# Calculate the error as per Barlow 4.3.4, and return its square root
	return float(np.sqrt(vals.T @ sigma @ vals))


def monte_carlo_error_propagation(expression, symbols, mu, sigma, N=5000):
	samples = np.random.multivariate_normal(mu, sigma, N).T
	expr = parse_expr(expression)
	f = sp.lambdify(symbols, expr, 'numpy')
	f_vec = np.vectorize(f)
	vals = f_vec(*samples)
	return vals


if __name__ == '__main__':
	expression = "m*v**2 / 2"
	symbols = "m v".split()
	mu = np.array([4, np.pi/4])
	sigma = np.array([[0.2, 0.1],[0.1, 0.4]])
	mont_vals = monte_carlo_error_propagation(expression, symbols, mu, sigma, N=100000)
	std = error_propagation(expression, symbols, mu, sigma)
	mean = float(parse_expr(expression).subs(zip(symbols, mu)))
	vals = np.random.normal(mean, std, 100000)


	fig, ax = plt.subplots()
	ax.hist(vals, histtype='step',label='Normal Error propagation')
	ax.hist(mont_vals, histtype='step', label='Monte Carlo')
	ax.legend()
	plt.show()

