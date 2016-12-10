# usage: [features, labels, labelIndices] = getTrainingInstances(classifier)

function [features, labels, labelIndices] = getTrainingInstances(classifier)
    
    features = [];
    labels = [];
    labelIndices = [];
    
    if(nargin != 1)
        print_usage();
    elseif(!isa(classifier, "classifier"))
        error("@classifier/getTrainingFeatures: requires classifier");
    endif
    
    features = classifier.trainingFeatures;
    labelIndices = classifier.trainingLabelInd;
    
    if(isargout(2))
        for i = 1:length(labelIndices)
            labels = [labels; (i-1) * ones(labelIndices(i), 1)];
        endfor
    endif

endfunction