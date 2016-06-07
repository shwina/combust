from combust.mixture import *
from combust.models.eos import IdealGasModel

import numpy as np
from numpy.testing import assert_allclose

def test_ideal_gas_model():
    cp = np.array([1005., 0., 1005.])
    cv = np.array([718., 0., 718.])
    R = cp-cv
    rho = np.ones([5, 5], dtype=np.float64)*1.187
    y = np.zeros([5, 5, 3], dtype=np.float64)
    y[:, :, 0] = 1.0
    u = np.zeros([5, 5], dtype=np.float64)
    v = np.zeros([5, 5], dtype=np.float64)
    E = rho*718.*298. +  0.5*rho*(u**2 + v**2)

    eos = IdealGasModel(cp, cv, R)
    T, P = eos.get_temperature_and_pressure(rho, y, E, u, v)
    assert_allclose(T, 298.)
