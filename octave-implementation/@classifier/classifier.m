# usage: obj = classifier(features, labels, labelNum)

function obj = classifier(features, labels, labelNum)
    
    obj = [];
    
    if(nargin == 3)
        if(!ismatrix(features) || !isvector(labels) || !isscalar(labelNum))
            error("@classifier/classifier: requires matrix, vector, scalar or no params");
        elseif(rows(features) != length(labels))
            error("@classifier/classifier: uneven number of features and labels");
        endif
    elseif(nargin != 0)
        print_usage();
    endif
    
    obj.trainingFeatures = zeros(0,0);
    obj.trainingLabelInd = {};
    obj.labelNum = 0;
    
    obj = class(obj, "classifier");
    
    if(nargin == 3)
        obj = setTrainingData(obj, features, labels, labelNum);
    endif
    
endfunction