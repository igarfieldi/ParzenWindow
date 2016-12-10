# usage: classifier = addInstance(classifier, feature, label)

function classifier = addInstance(classifier, feature, label)

    if(nargin != 3)
        print_usage();
    elseif(!isa(classifier, "classifier") || !ismatrix(feature) || !isvector(label))
        error("@classifier/addInstance: requires classifier, matrix, vector");
    endif
    
    if((label+1) > length(classifier.trainingLabelInd))
        classifier.trainingLabelInd = [ret.trainingLabelInd,...
                                repmat(classifier.trainingLabelInd(end),...
                                1, label - length(classifier.trainingLabelInd) + 1)];
    endif
    
    
    classifier.trainingFeatures = [classifier.trainingFeatures(1:classifier.trainingLabelInd(...
                                                                label+1), :);
                            feature;
                            classifier.trainingFeatures((classifier.trainingLabelInd(...
                                                                label+1)+1):end, :)];
    classifier.trainingLabelInd(label+1:end)++;

endfunction