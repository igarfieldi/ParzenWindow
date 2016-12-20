import numpy as np
from scipy.linalg import sqrtm
import math

class KernelDensityEstimator:
    def __init__(self, classes):
        self.trainingFeatures = [[] for i in range(classes)];

    def addTrainingInstance(self, instance, label):
        """
        Adds a single instance with its given label to the training set.
        :param instance: Feature vector
        :param label: Label of the instance
        """
        self.trainingFeatures[label].append(instance);


    def addTrainingInstances(self, instances, labels):
        """
        Adds multiple instances with their labels as a list to the training set.
        :param instances: List of instances
        :param labels: List of corresponding labels
        """
        for i in range(len(labels)):
            self.trainingFeatures[labels[i]].append(instances[i]);

    def classify(self, instances, bandwidth, kernel, bandwidthEstimator):
        """
        Computes the class probabilities for a given set of instances.
        The probabilities will be returned as a matrix with each class and instance combination.
        For the computation, the stored training instances will be used.

        :param instances: List of instances (ie. vectors of numeric features)
        :return: Matrix containing the class probabilities
        """
        priors = [0] * len(self.trainingFeatures);
        probabilities = [[0 for x in range(len(priors))] for y in range(len(instances))];

        # For all class labels estimate the prior of the class
        for i in range(len(self.trainingFeatures)):
            priors[i] = len(self.trainingFeatures[i]);

        # Now we iterate over all instances to determine the kernel densities
        for i in range(len(instances)):
            densities = [0] * len(self.trainingFeatures);

            # Then estimate the kernel density based on the stored instances with that label
            # TODO: variable kernel, incorporate bandwidth!
            for j in range(len(self.trainingFeatures)):
                densities[j] = estimateKernelDensity(instances[i], self.trainingFeatures[j], kernel, bandwidthEstimator);

            # Compute probabilities as product of estimated prior and density
            for j in range(len(densities)):
                probabilities[i][j] = priors[j]*densities[j];

            # Normalize probabilities
            probabilities[i] /= sum(probabilities[i]);

        return probabilities;

def kde(data):
    len = len(data)
    return 0

def estimateKernelDensity(instance, samples, kernel, bandwidthEstimator):
    """
    Returns the multivariate kernel density estimation for the given instance.
    The covariance for the kernel is estimated from the samples.
    :param instance: Instance to estimate density for
    :param samples: List of feature vectors to base estimation on
    :param kernel: Kernel to use for the estimation
    :param bandwidth: Smoothing matrix for the kernel
    :return: Estimated kernel density for the instance
    """

    # Get the covariance matrix from the samples
    cov = np.cov(samples, None, False);
    bandwidth = bandwidthEstimator(samples, cov, kernel);
    bandwidthInvSqrt = sqrtm(np.linalg.inv(bandwidth));
    bandwidthDet = np.linalg.det(bandwidthInvSqrt);

    density = 0;

    # Normal density estimation: for every sample compute kernel and sum up
    for i in range(0, np.shape(samples)[0]):
        density += kernel(np.dot(bandwidthInvSqrt, instance - samples[i]), cov) * bandwidthDet;
    density /= len(samples);

    return density;

def gaussKernel(x, cov):
    return np.exp(-0.5 * np.dot(np.dot(x, np.linalg.inv(cov)), x.transpose())) /\
           math.sqrt(math.pow(2.0*math.pi, x.size) * np.linalg.det(cov));

def estimateBandwidthSilvermanGauss(samples, cov, kernel):
    d = cov.shape[0];
    n = len(samples);
    bandwidth = np.zeros(cov.shape);

    for i in range(d):
        bandwidth[i][i] = math.pow(4.0/(d + 2.0), 1.0/(d+4.0)) * math.pow(n, -1.0/(d+4.0)) * cov[i][i];

    return bandwidth;

trainingData = [np.array([0, 0]),
                np.array([0, 1]),
                np.array([1, 0]),
                np.array([1, 0.5]),
                np.array([0.5, 1]),
                np.array([1, 1])];
trainingLabels = [0, 0, 0, 1, 1, 1];
testData = [np.array([0, 0]), np.array([0.5, 0.5]), np.array([1, 1])];

estimator = KernelDensityEstimator(2);
estimator.addTrainingInstances(trainingData, trainingLabels);
print(estimator.classify(testData, np.array([[1, 0], [0, 1]]), gaussKernel, estimateBandwidthSilvermanGauss));