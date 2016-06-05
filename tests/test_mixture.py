import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from combust.mixture import *

def test_mixture_molecular_wt():
    M = np.array([31.99, 31.99])
    y = np.ones([5,5,5,2])
    x = mole_fractions(y, M)
    assert_almost_equal(mixture_molecular_weight(x, M), 31.99)

def test_molar_volume():
    M = np.array([31.99]) 
    y = np.random.rand(3, 5, 7, 1)
    rho = np.random.rand(3, 5, 7)
    x = mole_fractions(y, M)
    Mm = mixture_molecular_weight(x, M)
    V = molar_volume(Mm, rho)
    assert_allclose(V, 31.99/rho, 0.1)

def test_molar_volume_two_species():
    M = np.array([31.99, 31.99])
    y = np.random.rand(3, 5, 7, 2)
    rho = np.random.rand(3, 5, 7)
    x = mole_fractions(y, M)
    Mm = mixture_molecular_weight(x, M)
    V = molar_volume(Mm, rho)
    assert_allclose(V, 31.99/rho, 0.1)

def test_mole_fractions_two_species():
    M = np.array([31.99])   
    y = np.ones([4, 4, 4, 2])
    x = mole_fractions(y, M)
    assert_almost_equal(mole_fractions(y, M), 0.5)
