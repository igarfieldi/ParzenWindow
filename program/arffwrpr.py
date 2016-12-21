from scipy.io.arff import loadarff


class ArffWrapper:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.data, self.meta = loadarff(f)


