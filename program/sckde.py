import numpy as np
import fourier
import floodfill
import math
import numpy.fft as fft

class SelfConsistentKDE:
    def __init__(self, data, samplesPerSigma):
        """
        This class represents a self-consistent, multivariate kernel density estimator as described by
        Bernacchia et al 2013 and (as an extension) O'Brien et al 2016.
        :param data: input data for kernel estimation
        :param samplesPerSigma: Density of sample points per standard deviation in each dimension
        """
        self.data = data
        self.dims = np.shape(data)[0]
        self.instances = np.shape(data)[1]
        self.ecf = np.zeros([1])

        # Compute necessary data statistics
        self.xMin = np.amin(data, 1)
        self.xMax = np.amax(data, 1)
        self.midPoint = 0.5 * (self.xMin + self.xMax)

        # Double the data range to avoid wrapping-around in frequency space!
        self.xMin += self.xMin - self.midPoint
        self.xMax += self.xMax - self.midPoint

        # Keep the new data range between -pi and +pi
        self.dataNorm = (self.xMax - self.xMin) / math.pi

        # Set the number of points for each dimensions
        # Fourier transformation needs powers of two so we take the next larger one
        numOfSigmas = (self.xMax - self.xMin) / np.std(data, axis=1)
        numXPoints = np.array([int(2 ** (math.ceil(math.log(ns * samplesPerSigma, 2)))) + 1 for ns in numOfSigmas])

        # Compute frequency sample points per axis/dimension
        self.axes = [np.linspace(xmin, xmax, numPoints) for xmin, xmax, numPoints in zip(self.xMin, self.xMax, numXPoints)]

        # Compute the sample grid in frequency space
        self.frequencyGrids = [fourier.transformFromDataToFrequencySpace((xg - av) / sd) for xg, av, sd in zip(self.axes, self.midPoint, self.dataNorm)]

        # Compute the ecf of the data
        self.computeEmpiricalCharacteristicFunction()
        # Apply the frequency filter and compute the self-consistent kernel estimate
        self.applyFrequencyFilter()
        # Compute the pdf and transform the kernel back to data space
        self.computeTransformedKernel()


    def computeEmpiricalCharacteristicFunction(self):
        # Compute the empirical characteristic function of the data
        # This is done by evaluating the discrete fourier transform at the provided frequency points

        # Get the maximum frequency grid length
        ntmax = np.amax([len(freqGrid) for freqGrid in self.frequencyGrids])

        # Since it is possible that not equal amounts of frequency points per dimension were provided we need a filler value
        # that tells the fourier transform to ignore this one (for performance reasons!)
        fillValue = -1e20
        filledFreqGrids = fillValue * np.ones([self.dims, ntmax])
        for v in range(self.dims):
            filledFreqGrids[v, :len(self.frequencyGrids[v])] = self.frequencyGrids[v]

        # Perform the fourier transform
        self.ecf = fourier.discreteFourierTransform(self.data, np.ones([self.instances], dtype=np.complex128), filledFreqGrids)

        # Normalize the ECF
        midPointAccessor = tuple([int((len(freqGrid) - 1) / 2) for freqGrid in self.frequencyGrids])
        if self.ecf[midPointAccessor] > 0.0:
            self.ecf = self.ecf/self.ecf[midPointAccessor]

    def applyFrequencyFilter(self):
        # Apply the filter as described in Bernacchia et al 2011
        N = float(self.instances)

        # Threshold for the search of a fitting hypervolume
        ecfThreshold = 4.0 * (N - 1.0) / (N * N)
        ecfSqr = abs(self.ecf) ** 2
        #print(ecfThreshold, ecfSqr)

        # Find all hypervolumes in the squared ecf which are above the threshold
        contiguousHypervolumes = floodfill.floodfill(ecfSqr, ecfThreshold)

        # Sort them by their distance to the centre
        sortedHypervolumes = floodfill.sortByDistanceFromCenter(contiguousHypervolumes, np.shape(ecfSqr))
        #print("Sorted hypervolumes", sortedHypervolumes)

        # TODO: flexible amount of hypervolumes!
        iCalcPhi = sortedHypervolumes[0]

        #print("Hypervolume to use", iCalcPhi)

        # Optimal kernel estimate
        kappaSC = (1.0 + 0.0j) * np.zeros(np.shape(self.ecf))

        # Fourier transform of optimal kernel
        self.phiSC = (0.0 + 0.0j) * np.zeros(np.shape(self.ecf))

        # Compute the fourier transform of optimal kernel; only for the indices within the hypervolume
        # This is equivalent to I_A (the indicator function for frequencies)
        for index in iCalcPhi:
            kappaSC[index] = (N / (2 * (N - 1))) \
                            * (1 + math.sqrt(1 - ecfThreshold / ecfSqr[index]))
            self.phiSC[index] = self.ecf[index] * kappaSC[index]

        #print("Kernel in Fourier space", kappaSC)
        #print("ECF", self.ecf)
        #print("Kernel estimate?\n", self.phiSC)

    def computeTransformedKernel(self):
        # Transform the kernel back to real space
        deltaT = np.array([freqGrid[2] - freqGrid[1] for freqGrid in self.frequencyGrids])
        self.pdf = fft.fftshift(fft.fftn(fft.ifftshift(self.phiSC)).real) * np.prod(deltaT) * (1. / (
        2 * math.pi)) ** self.dims

        # Normalize pdf
        self.pdf /= np.prod((self.xMax - self.xMin) / math.pi)

        # TODO: shift it to positive values only!

    def estimateDensity(self, instance):
        deltaT = np.array([freqGrid[2] - freqGrid[1] for freqGrid in self.frequencyGrids])
        estimate = fft.fftshift(fft.fftn(fft.ifftshift(self.phiSC)).real) * np.prod(deltaT) * (1. / (
        2 * math.pi)) ** self.dims