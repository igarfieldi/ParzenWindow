1;

addpath('kernelEstimation');

trainingFeatures = [0, 0;
					0, 1;
					1, 0;
					1, 1];
testFeatures = [0.25, 0.25;
				0.75, 0.75;
				0.925, 0.625;
				0.5, 0.5];
trainingLabels = [0; 0; 0; 1];

classifier = parzenWindowClassifier();
classifier = setTrainingData(classifier, trainingFeatures, trainingLabels, 2);

classifyInstances(classifier, testFeatures)