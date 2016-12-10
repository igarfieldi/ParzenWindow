# usage: sigma = getSigma(pwClassifier)

function sigma = getSigma(pwClassifier)

    sigma = [];
    
    if(nargin != 1)
        print_usage();
    elseif(!isa(pwClassifier, "parzenWindowClassifier"))
        error("@parzenWindowClassifier/getSigma: requires parzenWindowClassifier");
    endif
    
    sigma = pwClassifier.sigma;

endfunction