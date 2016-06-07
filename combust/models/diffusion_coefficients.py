from combust.constants import *
import numpy as np

class DiffusionCoefficientsModel:
    """ Model for the 'diffusion coefficients':
    measures of the diffusivity of each species
    into every other species. The diffusion
    coefficients are arranged in a
    square matrix of dimension "nspecies". `D_ij`
    describes the diffusivity of species `i`
    into species `j`. It follows that the matrix

    1. Is symmetric
    2. Has a diagonal of zeros
    """

    def __init__(self, M, adv, Tcrit, Pcrit):
        self.M = M
        self.adv = adv
        self.Tcrit = Tcrit
        self.Pcrit = Pcrit
        self._compute_matrices()

    def diffusion_coefficients_matrix(self, T, P, y):
        Tc_mix = np.tensordot(y, self.Tcrit, 1)
        Pc_mix = np.tensordot(y, self.Pcrit, 1)
        Tr = T/Tc_mix
        Pr = P/Pc_mix

        # high-pressure model constants:
        a = (Tr - 2.4)/1.5
        b = 6.293*Tr**2 - 9.0433*Tr + 2.9334
        c = 0.015*Tr - 0.036

        curve_fit_values = (
            (Tr < 2.4)*(np.exp(a*Pr) + b)/(1.0 + b) + 
            (Tr >= 2.4)*(1.0 + c*Pr))

        D = np.tensordot(curve_fit_values*(PATM/P)*(T**1.75)*1e-4, self.Dmat, 0)
        return D
    
    def _compute_matrices(self):
        M = self.M
        adv = self.adv
        M_mat = (2./(np.outer(1./M, np.ones_like(M)) + np.outer(np.ones_like(M), 1./M)))**0.5
        adv_mat = (np.outer(adv, np.ones_like(adv))**(1./3) + np.outer(np.ones_like(adv), adv)**(1./3))**2.0
        Dmat = (0.00143/(PATM*M_mat*adv_mat/1e5))*(1.0 - np.eye(M_mat.shape[0]))
        self.Dmat = Dmat

