from __future__ import division
import math
import numpy as np
import scipy as sp
from numpy import array
from scipy.integrate import ode

class InputOfCalls(object):
	"""
	Save the different calls to functions in order to see what might go wrong somewhere

	"""

	instances = {}

	def __init__(self, f):
		self.f = f
		self.inputcontents = []
		self.instances[f] = self

	def __call__(self, *args, **kwargs):
		print args
		print kwargs
		#self.inputcontents.append((*args, **kwargs))
		return self.f(*args, **kwargs)

	@classmethod
	def inputs(cls):
		"""return a dict of {function: [inputs],...}
		
		"""
		return dict([(f.__name__, cls.instances[f].inputcontents) for f in cls.instances])
