import numpy as np
from scipy.stats import truncnorm

def discrete_sampling(weights):
    agg_weights = np.cumsum(weights) / np.sum(weights)
    prob = np.random.rand()
    for i, item in enumerate(agg_weights):
        if prob < item:
            return i

# Switch a and b does not affect results
def uniform_sampling(a, b, all_levels=None):
    if all_levels is not None:
        jk = np.arange(all_levels + 1) / all_levels
    else:
        jk = np.random.uniform(low=0, high=1)
    F1 = a + (b - a) * jk
    return F1

# Switch a and b does not affect results
def expon_sampling(a, b, all_levels=None):
    if all_levels is not None:
        jk = np.arange(all_levels + 1) / all_levels
    else:
        jk = np.random.uniform(low=0, high=1)
    F1 = a * ((b / a) ** jk)
    return F1

# Half side of truncated gaussian distribution
# org is the center for spreading out, with highest probability.
def half_gauss_sampling(org, b, all_levels=None, spread=1.0):
    if all_levels is not None:
        jk = np.arange(all_levels + 1) / all_levels
    else:
        jk = np.random.uniform(low=0, high=1)
    if org < b:
        F1 = truncnorm.ppf(jk, 0, 1.0 / spread, loc=org, scale=spread * (b-org))
    else:
        F1 = truncnorm.ppf(jk, -1.0 / spread, 0, loc=org, scale=spread * (org - b))
    return F1

# Symmetric truncated gaussian distribution
# a, b are boundaries for truncation; the center of the distribution is the middle of the two
def gauss_sampling(a, b, all_levels=None, spread=1.0):
    if all_levels is not None:
        jk = np.arange(all_levels + 1) / all_levels
    else:
        jk = np.random.uniform(low=0, high=1)

    F1 = truncnorm.ppf(jk, -1.0 / spread, 1.0 / spread, loc=(a+b)/2.0, scale=spread * (b-a))
    return F1