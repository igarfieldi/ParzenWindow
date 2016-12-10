# usage: pwClassifier = setSigma(pwClassifier, sigma)

function pwClassifier = setSigma(pwClassifier, sigma)
    
    if(nargin != 2)
        print_usage();
    elseif(!isa(pwClassifier, "parzenWindowClassifier") || !isvector(sigma))
        error("@parzenWindowClassifier/setSigma: requires \
parzenWindowClassifier, vector");
    endif
    
    pwClassifier.sigma = sigma;

endfunction