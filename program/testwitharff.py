import kde
import sys
from arffwrpr import ArffWrapper

def arffEmulator(trainingPath, testPath, validationPath, **kwargs):
    trainingFile = ArffWrapper(trainingPath);
    testFile = ArffWrapper(testPath);
    validationFile = ArffWrapper(validationPath);

    dataset = (trainingFile.trainingData(), trainingFile.trainingLabels(),
               testFile.trainingData(), testFile.trainingLabels(),
               validationFile.trainingData(), validationFile.trainingLabels())

    return kde.do_stuff(dataset, **kwargs)

if __name__ == '__main__':
    args = sys.argv

    kdeKernel = 0
    estimator = 1
    shape = 1.0

    if(len(args) >= 5):
        kdeKernel = int(args[4])
    if(len(args) >= 6):
        estimator = int(args[5])
    if(len(args) >= 7):
        shape = float(args[6])

    result = arffEmulator(args[1], args[2], args[3], kdeKernel=kdeKernel, estimator=estimator, shape=shape)

    print('Test score: {}\nValidation score: {}\n{}').format(result['score_test'],
                                                             result['score_validation'],
                                                             result['message'])