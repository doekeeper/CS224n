# This is a repeat of q2_gradcheck.py code in order understand how gradient check is implemented - good luck!

import numpy as np
import random

def gradcheck_naive(f,x):		# f and x are the 2 input variables
	# Gradient check for a function f.
	# Arguments:
	# f -- a function that takes a single argument (# I believe it means x in this case)
	# and outputs the cost and its gradients
	# x -- the point (numpy array) to check the gradient at
	# so f's outputs are both cost and gradient, given the value of x
	
	rndstate = random.getstate()			""" why random.getstate() is needed here? """
	random.setstate(rndstate)				""" why random.setstate() is needed here? """
	fx, grad = f(x)							# fx and grad are both outputs, representing "function cost"
											# and "gradient", respectively
	h = 1e-4								# h is epsilon
	# Iterate over all indexes in x
	it = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])	
		""" iterate over x (# x seems to be an array/vector in this case)
			what flags = ['multi_index'] means?
			what op_flags = ['readwrite'] means?
		"""
	while not it.finished:					# np.dniter.finished is one attribute of np.nditer 
		ix = it.multi_index					""" ix is used to store index during iteration? """
		
		### MY CODE HERE:
		a = x[ix] + h
		b = x[ix] - h
		numgrad = (f(a)[0]-f(b)[0])/(2*h)
		### END MY CODE
		
		# Compare gradients
		reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))		# compare difference between numgrad and grad
		if reldiff > 1e-5:					# if difference is larger than 1e-5, print the following message as return
			print ("Gradient check failed.")	
			print ("First gradient error found at index %s" % str(ix))	# print index in which gradient check fails
			print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))	# print gradient results from numgrad and grad
			return
		it.iternext() """ step to next dimension? """		
	print ("Gradient check passed!")
	

def sanity_check():
	quad = lambda x: (np.sum(x**2), x*2)
	print ("running sanity checks...")
	gradcheck_naive(quad, np.array(123,456))	#scalar test
	gradcheck_naive(quad, np.random.randn(3,))	""" random 1-D vector generated here - is it related to getstate() or setstate()? """
	gradcheck_naive(quad, np.random.randn(4,5))	""" randowm 2-D array generated here - is it related to getstate() or setstate()? """
	print("")