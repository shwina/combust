from combust.models.spheat import *
import numpy as np
from numpy.testing import assert_allclose, assert_equal

def test_Model1():
    mwt = 1.0
    c1 = np.ones(5)
    c2 = np.ones(5)
    sph_model = Model1(mwt, c1, c2)

    T = 200.0
    assert_allclose(sph_model(T),
        Submodel1(c1, T)/mwt)
    
    T = 300.0
    assert_allclose(sph_model(T),
        Submodel3(c1, T)/mwt)

def test_Model2():
    mwt = 1.0
    c1 = np.ones(5)
    c2 = np.ones(5)
    sph_model = Model2(mwt, c1, c2)

    T = 200.0
    assert_allclose(sph_model(T),
        Submodel1(c1, T)/mwt)
    
    T = 500.0
    assert_allclose(sph_model(T),
        (Rgas/1000.0)*Submodel2(c1, T)/mwt)

    T = 1200.0
    assert_allclose(sph_model(T),
        (Rgas/1000.0)*Submodel2(c2, T)/mwt)

def test_Model3():
    mwt = 1.0
    c1 = np.ones(5)
    c2 = np.ones(5)
    sph_model = Model3(mwt, c1, c2)

    T = 200.0
    assert_allclose(sph_model(T),
        Submodel1(c1, T)/mwt)
    
    T = 500.0
    assert_allclose(sph_model(T),
        Submodel3(c1, T)/mwt)

    T = 1200.0
    assert_allclose(sph_model(T),
        Submodel3(c1, T)/mwt)

