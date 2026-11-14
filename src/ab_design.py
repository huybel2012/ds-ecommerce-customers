"""Power approximation for a future 1:1 randomized customer-level A/B test.

This calculator uses assumed rates, NOT experimental data in the Kaggle CSV.
"""
from math import ceil, sqrt
from scipy.stats import norm


def required_per_arm(p0:float, p1:float, alpha:float=.05, power:float=.8)->int:
    if not (0<p0<1 and 0<p1<1 and 0<alpha<1 and 0<power<1) or p0==p1:
        raise ValueError('Rates must be inside (0,1) and treatment must differ')
    avg=(p0+p1)/2
    za=norm.ppf(1-alpha/2)
    zb=norm.ppf(power)
    n=((za*sqrt(2*avg*(1-avg))+zb*sqrt(p0*(1-p0)+p1*(1-p1)))**2)/(p1-p0)**2
    return ceil(n)
