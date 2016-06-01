from combust.models.spheat import *
from numpy.testing import assert_allclose, assert_equal

def test_Model1():
    T = 200.0
    mwt = 1.0
    c1 = np.ones(5)
    c2 = np.ones(5)
    model = Model1(mwt, c1, c2)
    assert_allclose(model(T),
            Submodel1(c1, T)/mwt)
