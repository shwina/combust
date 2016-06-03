def mole_fractions(y, M):
    return ((y/M).T/((y/M).sum(axis=-1)).T).T

def mixture_molecular_weight(x, M):
    return (x*M).sum(axis=-1)

def molar_volume(Mm, rho):
    return Mm/rho
