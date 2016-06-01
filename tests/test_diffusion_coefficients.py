import sys
sys.path.append('..')
sys.path.append('../../../')

from theano import function, shared, sandbox
from theano.tensor import sqrt, tensordot, max
import numpy as np
from numpy.testing import *

from mixture import *
from diffusion_coefficients import DiffusionCoefficientsModel

""" Test the temperature solve algorithm
(Newton's method)

Species: Oxygen, Dodecane, Carbon-monoxide
Temperature (guess): 700.0
Density: 22.487
Mass fractions: [0.8268, 0.1731, 1.1145e-7]
Internal energy: -133529.59
Temperature corresponding to internal energy: 636.459 (test this)
"""

M = np.array([31.99, 168., 28.01])
adv = np.array([16.3, 250.86, 18.0])
Tcrit = np.array([154.6, 658.2, 132.9])
Pcrit = np.array([50.4e5, 18.2e5, 35.0e5])
Vcrit = np.array([0.073, 0.713, 0.0932])
Zcrit = np.array([0.288, 0.24, 0.295])
omega = np.array([0.025, 0.575, 0.066])

y = shared(np.array([0.8268, 0.1731, 1.1145e-007])*np.ones([16, 16, 16, 3]))
T = shared(np.ones([16, 16, 16])*636.45)
P = shared(np.ones([16, 16, 16])*3216669.38)

diff_coeff = DiffusionCoefficientsModel(M, adv, Tcrit, Pcrit)
D = diff_coeff.diffusion_coefficients_matrix(T, P, y)

f = function([], D)

def test_diffusion_coefficient_matrix():
    diffusion_coefficients = f()
    assert_allclose(diffusion_coefficients[:, :, :, 0, 1], 6.269e-7, 0.1)
    assert_allclose(diffusion_coefficients[:, :, :, 0, 2], 2.473e-6, 0.1)
    assert_allclose(diffusion_coefficients[:, :, :, 1, 2], 6.508e-7, 0.1)
    assert_equal(diffusion_coefficients[:, :, :, range(3), range(3)], 0.0)