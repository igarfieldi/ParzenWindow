# In this project [TODO: input correct description]
# is implemented by Florian Bethe and Angelika Ophagen.
#
# To use the algorithm just have a look at the main function of this python file.

import numpy as np
from arffwrpr import ArffWrapper
from kernel import GaussKernel
from bandwidth import SilvermanBandwidthEstimator
from bandwidth import McMcBandwidthEstimator
from sckde import SelfConsistentKDE
import sys

# This function applies the VFDT algorithm on data. Inserted data is expected to be discretized (see trainingData
# parameter description). The VFDT algorithm is applied on training data and evaluation data. The classification of the
# data given for evaluation is printed at standard out after classification is done.
# One hint for the current frontend: Users should be aware of the discretization that is applied on the data before it
# is handed over to this function.
#
# pathTrainingData:   path to an ARFF file containing the training data
# pathTestData:       path to an ARFF file containing the test data
# pathValidationData: path to an ARFF file containing the validation data
# bandwidthEstimator: bandwidth estimator to use, integer, 0 = Markov Chain, 1 = Silverman
# priorsShape:        parameter controlling the shape of prior bandwidth distribution in case of MC bandwidth estimator
#
# return:   returns a list of class labels for each instance in the ARFF file in the same ordering as in the ARFF file.
def main( pathTrainingData, pathTestData, pathValidationData, bandwidthEstimator = 1, priorsShape=1 ):
    #Todo: check all parameters for plausibility
    trainingFile = ArffWrapper( pathTrainingData );
    testFile = ArffWrapper( pathTestData );
    validationFile = ArffWrapper( pathValidationData );

    # prepare the kernel for the kernel density estimation
    kernel = GaussKernel( np.cov( trainingFile.trainingData(), rowvar=False ) );
    # prepare the classifier for the kernel density estimation
    classifier = ParzenWindowClassifier( kernel, trainingFile.labelCount() );
    # add the training data to the classifier
    classifier.addTrainingInstances( trainingFile.trainingData(), trainingFile.trainingLabels() );

    # use given bandwidth estimator
    if bandwidthEstimator == 0:
        estimator = McMcBandwidthEstimator( dims=trainingFile.dimensions(), shape=priorsShape );
    else:
        estimator = SilvermanBandwidthEstimator( np.cov( trainingFile.trainingData(), rowvar=False ) );
    classifier.estimateBandwidths( estimator );

    # classify training, test and validation data
    trainLabels = [];
    for instance in trainingFile.trainingData():
        probs = classifier.classify( [instance] );
        trainLabels.append( probs.index( max(probs) ) );

    testLabels = [];
    for instance in testFile.trainingData():
        probs = classifier.classify( [instance] );
        testLabels.append( probs.index( max(probs) ) );

    validationLabels = [];
    for instance in validationFile.trainingData():
        probs = classifier.classify( [instance] );
        validationLabels.append( probs.index( max(probs) ) );

    return trainLabels, testLabels, validationLabels

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

if __name__ == '__main__':
    args = sys.argv
    bandwidthEstimator = 1
    shape = 1
    if(len(args) >= 5):
        bandwidthEstimator = args[4]
    if(len(args) >= 6):
        shape = args[5]
    print(main(args[1], args[2], args[3], bandwidthEstimator, shape))