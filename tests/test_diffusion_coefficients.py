import numpy as np
from numpy.testing import assert_allclose, assert_equal
from combust.models.diffusion_coefficients import DiffusionCoefficientsModel

M = np.array([31.99, 168., 28.01])
adv = np.array([16.3, 250.86, 18.0])
Tcrit = np.array([154.6, 658.2, 132.9])
Pcrit = np.array([50.4e5, 18.2e5, 35.0e5])
Vcrit = np.array([0.073, 0.713, 0.0932])
Zcrit = np.array([0.288, 0.24, 0.295])
omega = np.array([0.025, 0.575, 0.066])

y = np.array([0.8268, 0.1731, 1.1145e-007])*np.ones([16, 16, 16, 3])
T = np.ones([16, 16, 16])*636.45
P = np.ones([16, 16, 16])*3216669.38

diff_coeff = DiffusionCoefficientsModel(M, adv, Tcrit, Pcrit)
D = diff_coeff.diffusion_coefficients_matrix(T, P, y)

def test_diffusion_coefficient():
    assert_allclose(D[:, :, :, 0, 1], 6.269e-7, 0.1)
    assert_allclose(D[:, :, :, 0, 2], 2.473e-6, 0.1)
    assert_allclose(D[:, :, :, 1, 2], 6.508e-7, 0.1)
    assert_equal(D[:, :, :, range(3), range(3)], 0.0)

