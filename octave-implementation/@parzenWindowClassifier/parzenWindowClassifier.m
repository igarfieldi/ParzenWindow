# usage: obj = parzenWindowClassifier(features, labels, sigma)

function obj = parzenWindowClassifier(features, labels, sigma)

    obj = [];
    
    if((nargin != 0) && (nargin != 2) && (nargin != 3))
        print_usage();
    endif
    
    obj.sigma = 0.1;
    obj.classifier = [];
    clas = @classifier();
    
    obj = class(obj, "parzenWindowClassifier", clas);
    
    if(nargin >= 2)
        if(!ismatrix(features) || !isvector(labels))
            error("@parzenWindowClassifier/parzenWindowClassifier: requires matrix, \
vector, vector");
        endif
        
        obj = setTrainingData(obj, features, labels);
    endif
    if(nargin == 3)
        if(!isvector(sigma))
            error("@parzenWindowClassifier/parzenWindowClassifier: sigma has to vector");
        endif
        obj.sigma = sigma;
    endif

endfunction