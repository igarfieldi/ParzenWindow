import numpy as np
import math

def cexp(arg):
    return math.exp(arg.real) * (math.cos(arg.imag) + 1.0j * math.sin(arg.imag))

def combinationIndices(index, shape, dims):
    dimInds = np.zeros([dims], dtype=np.int)

    currSize = np.prod(shape)

    for i in xrange(dims):
        currSize /= shape[i]
        dimInds[i] = index / currSize
        index -= dimInds[i] * currSize

    return dimInds

def discreteFourierTransform(abscissas, ordinates, frequencyGrids):
    dims = np.shape(abscissas)[0]
    instances = np.shape(abscissas)[1]

    freqSizes = np.zeros([dims],dtype=np.int)
    for i in xrange(dims):
        freqSizes[i] = len(frequencyGrids[i])

    dft = np.zeros([np.prod(freqSizes)],dtype=np.complex128)
    for index in xrange(np.prod(freqSizes)):
        dimInds = combinationIndices(index, freqSizes, dims)

        currDft = 0.0j

        for i in xrange(instances):
            exponent = 0.0j

            for d in xrange(dims):
                exponent += abscissas[d, i] * frequencyGrids[d, dimInds[d]]

            currDft += cexp(1.0j * exponent) * ordinates[i]
        dft[index] = currDft


    return np.reshape(dft, tuple(freqSizes))

def transformFromDataToFrequencySpace(dataPoints):
    deltas = dataPoints[1] - dataPoints[0]
    return np.fft.fftshift(np.fft.fftfreq(len(dataPoints), deltas / (2 * np.pi)))

def transformFromFrequencyToDataSpace(frequencyPoints):
    deltas = frequencyPoints[1] - frequencyPoints[0]
    return np.fft.fftshift(np.fft.fftfreq(len(frequencyPoints), deltas / (2 * np.pi)))