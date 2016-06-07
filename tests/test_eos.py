from combust.mixture import *
from combust.models.eos import IdealGasModel

import numpy as np
from numpy.testing import assert_allclose

def test_ideal_gas_single_species():
    rho = np.ones([5, 5], dtype=np.float64)*1.187
    u = np.zeros([5, 5], dtype=np.float64)
    v = np.zeros([5, 5], dtype=np.float64)
    E = rho*718.*298. +  0.5*rho*(u**2 + v**2)
    y = np.zeros([5, 5, 3], dtype=np.float64)

    cp = np.array([1005., 0., 1005.])
    cv = np.array([718., 0., 718.])
    R = cp-cv
    y[:, :, 0] = 1.0

    eos = IdealGasModel(cp, cv, R)
    T, P = eos.get_temperature_and_pressure(rho, y, E, u, v)
    assert_allclose(T, 298.)
    assert_allclose(P, 101519.362)

    cp = np.array([0., 1005., 1005.])
    cv = np.array([0., 718., 718.])
    R = cp-cv
    y[:, :, 0] = 0.0
    y[:, :, 1] = 1.0

    eos = IdealGasModel(cp, cv, R)
    T, P = eos.get_temperature_and_pressure(rho, y, E, u, v)
    assert_allclose(T, 298.)
    assert_allclose(P, 101519.362)

def test_ideal_gas_multiple_species():
    rho = np.ones([5, 5], dtype=np.float64)*1.187
    u = np.zeros([5, 5], dtype=np.float64)
    v = np.zeros([5, 5], dtype=np.float64)
    E = rho*718.*298. +  0.5*rho*(u**2 + v**2)
    y = np.zeros([5, 5, 3], dtype=np.float64)

    cp = np.array([1005., 1005., 1005.])
    cv = np.array([718., 718., 718.])
    R = cp-cv
    y[:, :, 0] = 0.5
    y[:, :, 1] = 0.5

    eos = IdealGasModel(cp, cv, R)
    T, P = eos.get_temperature_and_pressure(rho, y, E, u, v)
    assert_allclose(T, 298.)
    assert_allclose(P, 101519.362)
