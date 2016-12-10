# usage: classifier = setTrainingData(classifier, features, labels, labelNum)

function classifier = setTrainingData(classifier, features, labels, labelNum)
    
    if(nargin != 4)
        print_usage();
    elseif(!isa(classifier, "classifier") || !ismatrix(features)
                || !isvector(labels) || !isscalar(labelNum))
        error("@classifier/setTrainingData: requires classifier, matrix, vector, scalar");
    endif
    
    classifier.trainingFeatures = zeros(0, 0);
    classifier.trainingLabelInd = zeros(labelNum, 1);
    classifier.labelNum = labelNum;
    
    for i=1:length(labels)
        if((labels(i)+1) > length(classifier.trainingLabelInd))
            error("@classifier/setTrainingData: encountered label larger than labelNum");
        endif
        
        classifier.trainingFeatures = [classifier.trainingFeatures(1:...
                                            classifier.trainingLabelInd(labels(i)+1), :);
                                features(i, :);
                                classifier.trainingFeatures((...
                                    classifier.trainingLabelInd(...
                                                    labels(i)+1)+1):end, :)];
        classifier.trainingLabelInd(labels(i)+1:end)++;
    endfor

endfunction