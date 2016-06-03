from combust.constants import *
import numpy as np

class EOSModel:
    """ Peng-Robinson equation of state model
    for arbitrary number of species.
    Ref: Ph.D. Thesis, Sridhar Palle """

    def __init__(self, Tcrit, Pcrit, Vcrit, Zcrit, omega):
        """
        Parameters:
        -----------
        Tcrit: NumPy array of critical temperatures of the species
        Pcrit: NumPy array of critical pressures of the species
        Vcrit: NumPy array of critical volumes of the species
        Zcrit: NumPy array of critical compressibility factors of the species
        omega: NumPy array of acentric factors of the species
        """
        self.Tcrit = Tcrit
        self.Pcrit = Pcrit
        self.Vcrit = Vcrit
        self.Zcrit = Zcrit
        self.omega = omega
        
        # FIXME: may want to rename to initialize or something
        self._compute_critical_matrices()

    def internal_energy(self, T, V, x, h0m, cp0m, mmw):
        """ 
        Parameters:
        -----------
        T: Temperatures at each computational point
        V: Molar volumes
        x: Mole fractions of each species
        h0m: Low-pressure mixture enthalpy
        cp0m: Low-pressure mixture specific heat 
        mmw: Mixture molecular weight

        Returns:
        --------
        h, dhdT: Mixture internal energy and its derivative
                 at each computational point
        """
        Am, Bm, dAmdT, d2AmdT2 = self.mixing_constants(T, x)
        h = h0m + (-Rgas*(x.sum(axis=-1))*T
                    + (1./(2.0*np.sqrt(2.0)*Bm))*np.log(
                      (V + (1.0 - np.sqrt(2.0))*Bm)/(V + (1.0 +
                          np.sqrt(2.0))*Bm))*
                      (Am - T*dAmdT)
                   )/mmw
        dhdT = cp0m + (-Rgas*np.sum(x, axis=-1) +
                (1./(2.0*np.sqrt(2.0)*Bm))*np.log(
                    (V + (1.0 - np.sqrt(2.0))*Bm)/(V + (1.0 + np.sqrt(2.0))*Bm))*
                (-T*d2AmdT2))/mmw
        return h, dhdT

    def specific_heat(self, T, V, x, h0m, cp0m, mmw):
        """ 
        Parameters:
        -----------
        T: Temperatures at each computational point
        V: Molar volumes
        x: Mole fractions of each species
        h0m: Low-pressure mixture enthalpy
        cp0m: Low-pressure mixture specific heat 
        mmw: Mixture molecular weight

        Returns:
        --------
        cp: Mixture specific heat
            at each computational point
        """
        Am, Bm, dAmdT, d2AmdT2 = self.mixing_constants(T, x)
        dPdT = Rgas/(V - Bm)*x.sum(axis=-1) - dAmdT/(V**2.0 + 2*V*Bm - Bm**2)
        dPdV = -Rgas*T/(V - Bm)**2.0*x.sum(axis=-1)*(1.0 - (2*Am)/((Rgas*T)*(V + Bm)*(V/(V - Bm) + Bm/(V + Bm))**2.0))
        cp = cp0m + (-T*(dPdT**2./dPdV) - Rgas*x.sum(axis=-1) - (T*d2AmdT2/(2.0*np.sqrt(2.0)*Bm))*np.log(
                (V + (1.0 - np.sqrt(2.0))*Bm)/(V + (1.0 + np.sqrt(2.0))*Bm)))/mmw
        return cp

    def mixing_constants(self, T, x):
        """ Compute Am, Bm, dAmdT and d2AmdT2,
        the "mixing constants" of the model and
        associated derivatives."""

        Tcmat = self.Tcmat
        Pcmat = self.Pcmat
        Vcmat = self.Vcmat
        Omat = self.Omat
        C = 0.37464 + 1.54226*Omat - 0.26992*Omat**2.0
        B = 0.07796*Rgas*(np.diagonal(Tcmat)/
                np.diagonal(Pcmat))
        A = (0.457236*((Rgas*Tcmat)**2.0)/Pcmat)*\
                ((1.0 +  C*(1.0 -
                    np.sqrt(np.tensordot(T, 1./Tcmat, 0))))**2.0)

        Am = np.tensordot((A*x[:,:,:,:,None]*x[:,:,:,None,:]),
                np.ones(Tcmat.shape), 2)
        Bm = np.tensordot(x, B, 1)

        G = C*np.sqrt(np.tensordot(T, 1./Tcmat, axes=0))/(
                1.0 + C*(1.0 -
                np.sqrt(np.tensordot(T, 1./Tcmat, axes=0))))
        
        dAmdT = (-1./T)*(np.tensordot(
            G*A*x[:,:,:,None,:]*x[:,:,:,:,None],
            np.ones(Tcmat.shape),
                    2))
        d2AmdT2 = 0.457236*(Rgas**2)/(2.0*T*sqrt(T))*\
            np.tensordot(
                (C*(1.+C)*Tcmat*np.sqrt(Tcmat)/Pcmat)*
                x[:,:,:,None,:]*x[:,:,:,:,None],
                np.ones(Tcmat.shape),
                                2)
        return Am, Bm, dAmdT, d2AmdT2

    def _compute_critical_matrices(self):
        """ Construct matrices required in
        the model calculations """
        Tc = np.asarray(self.Tcrit)
        Pc = np.asarray(self.Pcrit)
        Vc = np.asarray(self.Vcrit)
        Zc = np.asarray(self.Zcrit)
        om = np.asarray(self.omega)
        N = len(Tc)
        Tcmat = np.sqrt(np.outer(Tc, Tc))
        Zcmat = 0.5*(np.outer(np.ones(N), Zc) +
                     np.outer(Zc, np.ones(N)))
        Omat  = 0.5*(np.outer(np.ones(N), om) +
                     np.outer(om, np.ones(N)))
        Vcmat = (1./8)*(np.outer(np.ones(N), Vc**(1./3)) +
                        np.outer(Vc**(1./3), np.ones(N)))**3.0
        Pcmat = Zcmat*Tcmat*Rgas/Vcmat
        np.fill_diagonal(Pcmat, Pc)
        self.Tcmat = Tcmat
        self.Pcmat = Pcmat
        self.Vcmat = Vcmat
        self.Zcmat = Zcmat
        self.Omat = Omat
