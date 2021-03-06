import numpy as np
import scipy
import math
from kernel import GaussKernel
from mcmc import metropolisHastingsSampling

def estimatePrior(bandwidth, shape):
    """
    Computes the prior probability of the given bandwidth.
    :param bandwidth: Cholesky-decomposed bandwidth matrix (lower triangle)
    :param shape: Parameter controlling the shape of the priors
    :return: estimate for prior
    """
    prior = 1.0

    for b in bandwidth.flatten():
        prior *= 1.0 + shape * b

    return 1.0 / prior

def estimateLikelihood(kde, choleskyBandwidth):
    """
    Computes the likelihood of the data given the bandwidth.
    :param kde: Kernel density estimator which provides the LOO estimation
    :param choleskyBandwidth: decomposed version of the bandwidth matrix
    :return: estimated likelihood
    """
    likelihood = 1.0
    for sample in kde.getSamples():
        likelihood *= kde.leaveOneOutEstimate(sample, choleskyBandwidth)

    return likelihood

def randomWalk(theta_p):
    '''
    Computes a new proposed sample by adding a random vector (drawn from the multivariate standard distribution) to
    the old sample. Also truncates samples at zero
    :param theta_p: previous sample
    :return: new proposed sample
    '''
    nextSample = theta_p + np.random.normal(0, 1, (len(theta_p), len(theta_p))) * np.tril(np.ones(theta_p.shape))
    nextSample[nextSample < 0] = 0
    return nextSample

def proposedDistribution(oldState, newState):
    '''
    Computes the probability of state transition according to a (truncated) multivariate normal distribution.
    The factor of two is due to the truncation at zero; this keeps it a valid pdf.
    :param oldState: Old state of the markov chain
    :param newState: New (proposed) state of the markov chain
    :return: Probability of state transition
    '''
    return 2.0 * GaussKernel(np.identity(len(oldState[np.tril_indices(len(oldState))]))).eval(oldState[np.tril_indices(len(oldState))])

class McMcBandwidthEstimator:
    def __init__(self, dims, shape=1, priors=None, iterations=2500, burnIn=500, beVerbose=False):
        self.shape = shape
        if priors is None:
            self.priors = np.identity(dims, dtype=np.float64)
        else:
            if np.shape(priors) != (dims, dims):
                raise ValueError('The priors must have the same dimensionality as the bandwidth!')
            else:
                self.priors = priors
        self.iterations = max(1, iterations)
        self.burnIn = min(self.iterations, burnIn)
        self.beVerbose = beVerbose

    def estimateBandwidth(self, kde):
        # Target function (only proportional to what we really have as a distribution):
        # This is the product of the prior and the likelihood (see the paper for more detailed explanation)
        target = lambda theta: estimatePrior(theta, self.shape) * estimateLikelihood(kde, theta)

        bandwidths, accepted = metropolisHastingsSampling(self.priors, self.iterations, target, proposedDistribution,
                                                randomWalk)

        optimalBandwidth = np.zeros(self.priors.shape)

        # Attempt at computing the ergodic average
        for b in bandwidths[self.burnIn:]:
            optimalBandwidth += b
        optimalBandwidth /= float(self.iterations - self.burnIn)

        if self.beVerbose:
            print('Acceptance: {} out of {} ({}%)').format(accepted, self.iterations, 100.0 * accepted / float(self.iterations))
            print('Optimal bandwidth:\n{}').format(np.dot(optimalBandwidth, optimalBandwidth.transpose()))

        return optimalBandwidth

class SilvermanBandwidthEstimator:
    """
    Estimator for bandwidth using silverman's rule of thumb.
    """
    def __init__(self, cov):
        self.cov = cov

    def estimateBandwidth(self, kde):
        """
        Computes bandwidth estimate using the given covariance and kernel density estimator.
        :param kde: Density estimator. Only provides the samples to use
        :return: Estimated bandwidth matrix (diagonal only and cholesky decomposed)
        """
        d = self.cov.shape[0]
        n = len(kde.getSamples())

        bandwidth = np.zeros(self.cov.shape)

        for i in range(d):
            bandwidth[i][i] = math.pow(4.0 / (d + 2.0), 1.0 / (d + 4.0)) * math.pow(n, -1.0 / (d + 4.0)) * self.cov[i][i]

        return np.linalg.cholesky(bandwidth)
