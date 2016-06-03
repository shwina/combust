from combust.mixture import *
from combust.models.eos import EOSModel
from combust.models import enthalpy
from combust.models import spheat

import numpy as np
from numpy.testing import assert_allclose
import pytest

@pytest.fixture
def temperature():
    M = [31.99, 168., 28.01]
    Tcrit = [154.6, 658.2, 132.9]
    Pcrit = [50.4e5, 18.2e5, 35.0e5]
    Vcrit = [0.073, 0.713, 0.0932]
    Zcrit = [0.288, 0.24, 0.295]
    omega = [0.025, 0.575, 0.066]

    M = np.array(M)
    y = np.array([ 0.8268, 0.1731, 1.1145e-007])*np.ones([16, 16, 16, 3])
    T = np.ones([16, 16, 16])*700.0
    rho = np.ones([16, 16, 16])*22.4827
    cp0m = np.zeros([16, 16, 16])
    h0m = np.zeros([16, 16, 16])
    h = np.ones([16, 16, 16])*-133529.59
    tol = 1e-6

    species = [
                {
                'SpecificHeatModel': spheat.Model1(31.99, [2.811e4, -3.68e-3, 1.746e-2, -1.065e-5], 
                                                                [29.659e3, 6.137, -1.1865e-3, 0.0957e-6, -0.2196e9, -9.86e6, 0.0]),
                'EnthalpyModel': enthalpy.Model3(31.99, [2.811e4, -3.68e-3, 1.746e-2, -1.065e-5],
                                                              [29.659e3, 6.137, -1.1865e-3, 0.0957e-6, -0.2196e9, -9.86e6, 0.0])
                },
                {
                'SpecificHeatModel': spheat.Model2(168., [0.395e4, 0.10207e3, 0.1312e-1, -0.766492e-4, 0.345e-7, -0.5209e8],
                                                               [0.364e5, 0.546e2, -0.1609e-1, 0.2147e-5, -0.1013e-9, -0.639e8]),
                'EnthalpyModel': enthalpy.Model2(168., [0.395e4, 0.10207e3, 0.1312e-1, -0.766492e-4, 0.345e-7, -0.5209e8],
                                                             [0.364e5, 0.546e2, -0.1609e-1, 0.2147e-5, -0.1013e-9, -0.639e8])
                },
                {
                'SpecificHeatModel': spheat.Model3(28.01, [25.567e3, 6.0961, 4.054e-3, -2.67e-6, 0.131e9, -118.0089e6, -110.527e6],
                                                                [35.1507e3, 1.3, -0.2059e-3, 0.01355e-6, -3.2828e9, -127.83e6, -110.5271e6]),
                'EnthalpyModel': enthalpy.Model4(28.01,  [25.567e3, 6.0961, 4.054e-3, -2.67e-6, 0.131e9, -118.0089e6, -110.527e6],
                                                                [35.1507e3, 1.3, -0.2059e-3, 0.01355e-6, -3.2828e9, -127.83e6, -110.5271e6])
                }
            ]

    x = mole_fractions(y, M)
    Mm = mixture_molecular_weight(x, M)
    V = molar_volume(Mm, rho)

    eos = EOSModel(Tcrit, Pcrit, Vcrit, Zcrit, omega)

    for i, sp in enumerate(species):
        cp0m += y[:,:,:,i]*sp['SpecificHeatModel'](T)
        h0m += y[:,:,:,i]*sp['EnthalpyModel'](T)

    h_guess, dhdT_guess = eos.internal_energy(T, V, x, h0m, cp0m, Mm)

    for i in range(100):
        T = T - abs(h_guess-h)/dhdT_guess
        is_converged = np.max(np.abs(h_guess-h)) <= tol
        if is_converged:
            break

    return T

def test_test(temperature):
    assert_allclose(temperature, 636.459)
