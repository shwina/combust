from combust.constants import *

# To define a new model:
# 1. Add a Submodel (if necessary)
# 2. Add a Model by subclassing SpHeatModel or writing its own constructor

def Submodel1(c, T):
    return c[0]*T**0 + c[1]*T + c[2]*T**2 + c[3]*T**3

def Submodel2(c, T):
    return RGAS*(c[0]*T**0 + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4)

def Submodel3(c, T):
    return c[0]*T**0 + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]/(T**2)

class SpHeatModel:
    def __init__(self, molecular_weight, c1, c2):
        # default constructor accepts two sets of
        # coefficients; override for additional
        # coefficients
        self.molecular_weight = molecular_weight
        self.c1 = c1
        self.c2 = c2

class Model1(SpHeatModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        cp0 = ((T < 298.0)*
                Submodel1(c1, T)/self.molecular_weight +
              (T >= 298.0)*
                Submodel3(c2, T)/self.molecular_weight)
        return cp0

class Model2(SpHeatModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        cp0 = ((T < 300.0)*
                Submodel1(c1, T)/self.molecular_weight +
              ((T >= 300.0) & (T < 1000.0))*
                RGAS/1000.*Submodel2(c1, T)/self.molecular_weight +
               (T >= 1000.0)*
                RGAS/1000.*Submodel2(c2, T)/self.molecular_weight)
        return cp0

class Model3(SpHeatModel):
    def __call__(self, T):
        c1 = self.c1
        c2 = self.c2
        cp0 = ((T < 298.0)*
                Submodel1(c1, T)/self.molecular_weight +
              ((T >= 298.0) & (T < 1300.0))*
                Submodel3(c1, T)/self.molecular_weight +
               (T >= 1300.0)*
                Submodel3(c2, T)/self.molecular_weight)
        return cp0

# class Model4(SpHeatModel):

#     def __init__(self, c1, c2, c3):
#         self.c1 = c1
#         self.c2 = c2
#         self.c3 = c3

#     def __call__(self, T):
#         c1 = self.c1
#         c2 = self.c2
#         cp0 = ((T < 500.0)*
#                 (c1[0] + c1[1]*T + c1[2]*T**2.0 + c1[3]*T**3.0)/self.molecular_weight +
#                (T >= 500.0)*
#                 (c1[0]*T**0 + c1[1]*T + c1[2]*T**2.0 + c1[3]*T**3.0 + c1[4]/(T**2.0))/self.molecular_weight)
#         return cp0
