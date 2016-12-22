import numpy as np
from scipy.io.arff import loadarff


class ArffWrapper:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.data, self.meta = loadarff(f);

    def trainingData(self):
        """
        Returns the data from the ARFF file as numpy array of arrays,
        without the label information
        :rtype: array of arrays with instance values
        """
        # everything but the last column
        trainingData = self.data[self.meta.names()[:-1]];
        # converts the record array to a numpy array
        trainingData = trainingData.view(np.float)\
            .reshape(self.data.shape + (-1,));
        return trainingData;

    def trainingLabels(self):
        """
        Returns only the label data from the ARFF file, converted to integer as a numpy array
        :return: numpy array of integer label entries
        """
        trainingLabels = [];
        # only the last column
        for x in self.data[self.meta.names()[-1:]]:
            trainingLabels.append(np.int(x[0]));
        # converts the integer list to a numpy array
        trainingLabels = np.array(trainingLabels);
        return trainingLabels;

myArff = ArffWrapper('testdata/iris-testdata.arff');
print myArff.meta.names();
print myArff.meta.types();
print myArff.data;
print myArff.trainingLabels();
print myArff.trainingData();
