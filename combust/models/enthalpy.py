from combust.constants import *

# To define a new model:
# 1. Add a Submodel (if necessary)
# 2. Add a Model by subclassing EnthalpyModel or writing its own constructor

def Submodel1(c, T):
    return c[0]*(T-298.) + c[1]/2*(T**2-298.**2) + c[2]/3*(T**3-298.**3) + c[3]/4*(T**4-298.**4)

def Submodel2(c, T):
    return (c[0]*T + c[1]/2*T**2 + c[2]/3*T**3 + c[4]*T**4 - c[5]/T + c[6] - c[7])

def Submodel3(c, T):
     return (1.*c[0]*(T-298.) + 1./2*c[1]*(T**2-298.**2) + 
                 1./3*c[2]*(T**3-298.**3) + 1./4*c[4]*(T**3-298**4))

def Submodel4(c, T):
    return (c[0]*T**0 + c[1]*0.5*T + c[2]*(1./3)*T**2 + c[3]*(1./4)*T**3 + 
                    c[4]*(1./5)*T**4 + c[5]/T)

def Submodel5(c, T):
    return (c[0]*T + c[1]*T**2/2.0 + c[2]*T**3/3.0 + c[3]*T**4/4.+
               - c[4]/T + c[5]*T**0.0 - c[6]*T**0.0)

class EnthalpyModel:
    def __init__(self, molecular_weight, c1, c2):
        # default constructor accepts two sets of
        # coefficients; override for additional
        # coefficients
        self.molecular_weight = molecular_weight
        self.c1 = c1
        self.c2 = c2

class Model1(EnthalpyModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        h0 = ((T < 298.0)*
                Submodel1(c1, T)/self.molecular_weight +
              (T >= 298.0)*
                Submodel2(c2, T)/self.molecular_weight)
        return h0

class Model2(EnthalpyModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        h0 = ((T < 300.0)*
               Submodel3(c1, T)/self.molecular_weight  +
             ((T >= 300) & (T < 1000.0))*
                (Rgas/1000.*T*Submodel4(c1, T)/self.molecular_weight) +
              (T >= 1000.0)*
                (Rgas/1000.*T*
                    Submodel4(c2, T))/self.molecular_weight)
        return h0

class Model3(EnthalpyModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        h0 = ((T < 298.0)*
                Submodel1(c1, T)/self.molecular_weight +
              (T >= 298.0)*
                Submodel5(c2, T)/self.molecular_weight)
        return h0

class Model4(EnthalpyModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        h0 = ((T < 298.0)*
                Submodel1(c1, T)/self.molecular_weight +
             ((T >= 298.0) & (T < 1300.0))*
                Submodel5(c1, T)/self.molecular_weight +
              (T >= 1300.0)*
                Submodel5(c2, T)/self.molecular_weight)
        return h0
