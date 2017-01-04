import numpy as np
import scipy.stats as st

def metropolisHastingsSampling(parameters, iterations, target, proposed, proposedSampler):
    """
    Performs Metropolis-Hastings random walk.
    :param parameters: array of parameters to get samples for
    :param iterations: number of iterations to perform
    :param target: target function with proportional distribution to posterior
    :param proposed: proposed distribution to sample from
    :param step: distribution to draw step size from
    :return: list of samples for parameters (it is wise to discard an initial percentage!), acceptance rate of samples
    """
    samples = [np.zeros(len(parameters))] * (iterations + 1);
    samples[0] = parameters;

    # Tracks the number of accepted samples
    acceptance = 0;

    for i in range(iterations):
        # Get new proposed sample from sampler
        parameters_p = proposedSampler(parameters);
        # Compute acceptance probability (for explanation of formula see Metropolis-Hastings algorithm)
        rho = min(1, target(parameters_p) * proposed(parameters, parameters_p) /
                  (target(parameters) * proposed(parameters_p, parameters)));

        # If we draw lower than acceptance we accept, else discard the sample
        u = np.random.uniform();
        if u < rho:
            parameters = parameters_p;
            acceptance += 1;

        # Regardless of whether we accepted the sample or not, the current sample point still counts as an element
        # in the resulting markov chain
        samples[i+1] = parameters;

    return samples, acceptance / float(iterations);