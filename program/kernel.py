import numpy as np
import math

class GaussKernel:
    def __init__(self, cov):
        self.invCov = np.linalg.inv(cov)
        self.detCov = np.linalg.det(cov)

    def eval(self, x):
        return np.exp(-0.5 * np.dot(np.dot(x, self.invCov), x.transpose())) /\
               math.sqrt(math.pow(2.0*math.pi, x.size) * self.detCov)