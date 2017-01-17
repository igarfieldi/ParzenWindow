import numpy as np
import scipy.stats as sp
import math

class GaussKernel:
    def __init__(self, cov):
        self.cov = cov

    def eval(self, x):
        return sp.multivariate_normal.pdf(x, mean=np.zeros(x.shape), cov=self.cov)

class EpanechnikovKernel:
    def __init__(self, dims):
        self.dims = dims

    def eval(self, x):
        res = math.pow(0.75, self.dims)
        for d in range(self.dims):
            if(abs(x[d]) <= 1):
                res *= (1.0 - x[d]*x[d])
        return res

class PicardKernel:
    def __init__(self, dims):
        self.dims = dims

    def eval(self, x):
        res = math.pow(0.5, self.dims)
        for d in range(self.dims):
            res *= math.exp(-abs(x[d]))
        return res
