import numpy as np
from scipy.linalg import sqrtm
import math
import functools
import scipy.stats as st
from arffwrpr import ArffWrapper
from mcmc import metropolisHastingsSampling
from kernel import GaussKernel
from bandwidth import SilvermanBandwidthEstimator
from bandwidth import McMcBandwidthEstimator

class KernelDensityEstimator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.samples = []

    def setKernel(self, kernel):
        self.kernel = kernel

    def addSample(self, sample):
        self.samples.append(sample)

    def addSamples(self, samples):
        for sample in samples:
            self.samples.append(sample)

    def getSamples(self):
        return self.samples

    def leaveOneOutEstimate(self, instance, choleskyBandwidth):
        bandwidthDet = np.linalg.det(choleskyBandwidth)

        density = 0

        # Normal density estimation: for every sample compute kernel and sum up
        for sample in self.samples:
            if (sample != instance).any():
                density += self.kernel.eval(np.dot(choleskyBandwidth, instance - sample)) * bandwidthDet
        density /= len(self.samples) - 1.0

        return density

    def estimateDensity(self, instance, choleskyBandwidth):
        bandwidthDet = np.linalg.det(choleskyBandwidth)

        if len(self.samples) == 0:
            return 0

        density = 0

        # Normal density estimation: for every sample compute kernel and sum up
        for i in range(0, np.shape(self.samples)[0]):
            density += self.kernel.eval(np.dot(choleskyBandwidth, instance - self.samples[i])) * bandwidthDet
        density /= len(self.samples)

        return density

class ParzenWindowClassifier:
    def __init__(self, kernel, classes):
        self.estimators = []
        # TODO: different kernels for different classes!
        for i in range(classes):
            self.estimators.append(KernelDensityEstimator(kernel))

    def addTrainingInstance(self, instance, label):
        """
        Adds a single instance with its given label to the training set.
        :param instance: Feature vector
        :param label: Label of the instance
        """
        self.estimators[label].addSample(instance)


    def addTrainingInstances(self, instances, labels):
        """
        Adds multiple instances with their labels as a list to the training set.
        :param instances: List of instances
        :param labels: List of corresponding labels
        """
        for i in range(len(labels)):
            self.estimators[labels[i]].addSample(instances[i])

    def estimateBandwidths(self, estimator):
        self.bandwidths = []
        for e in self.estimators:
            self.bandwidths.append(estimator.estimateBandwidth(e))

    def classify(self, instances):
        """
        Computes the class probabilities for a given set of instances.
        The probabilities will be returned as a matrix with each class and instance combination.
        For the computation, the stored training instances will be used.

        :param instances: List of instances (ie. vectors of numeric features)
        :return: Matrix containing the class probabilities
        """

        priors = [0.0] * len(self.estimators)
        probabilities = [[0.0 for x in range(len(priors))] for y in range(len(instances))]

        # For all class labels estimate the prior of the class
        for i in range(len(self.estimators)):
            priors[i] = len(self.estimators[i].getSamples())

        # Now we iterate over all instances to determine the kernel densities
        for i in range(len(instances)):
            densities = [0] * len(self.estimators)

            # Then estimate the kernel density based on the stored instances with that label
            # TODO: variable kernel, incorporate bandwidth!
            for j in range(len(self.estimators)):
                densities[j] = self.estimators[j].estimateDensity(instances[i], self.bandwidths[j])

            # Compute probabilities as product of estimated prior and density
            for j in range(len(densities)):
                probabilities[i][j] = priors[j]*densities[j]

            # Normalize probabilities
            probabilities[i] /= sum(probabilities[i])

        return probabilities

from sckde import SelfConsistentKDE

testFile = ArffWrapper('testdata/iris-testdata.arff');

training = np.random.permutation(len(testFile.trainingData()))
test = training[:len(training)/2]
training = training[len(training)/2+1:]

# TODO: fix wrong transposition!
#selfConsistent = SelfConsistentKDE(testFile.trainingData().transpose(), 1)
#print(selfConsistent.phiSC)

# TODO: bandwidth estimate must not be below zero!
classifier = ParzenWindowClassifier(GaussKernel(np.cov(testFile.trainingData()[training], rowvar=False)), testFile.labelCount())
classifier.addTrainingInstances(testFile.trainingData()[training], testFile.trainingLabels()[training])
classifier.estimateBandwidths(McMcBandwidthEstimator(dims=testFile.dimensions(), beVerbose=True))
#classifier.estimateBandwidths(SilvermanBandwidthEstimator(np.cov(testFile.trainingData()[training], rowvar=False)))

acc = 0
total = 0

for i in range(len(test)):
    probs = classifier.classify([testFile.trainingData()[test[i]]])

    if(probs.index(max(probs)) == testFile.trainingLabels()[test[i]]):
        acc += 1
    total += 1

print(acc / float(total))