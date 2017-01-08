import numpy as np
import math

class GaussKernel:
    def __init__(self, cov):
        self.invCov = np.linalg.inv(cov)
        self.detCov = np.linalg.det(cov)

    def eval(self, x):
        return np.exp(-0.5 * np.dot(np.dot(x, self.invCov), x.transpose())) /\
               math.sqrt(math.pow(2.0*math.pi, x.size) * self.detCov)

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