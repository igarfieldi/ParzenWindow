# usage: pwClassifier = estimateSigma(pwClassifier, features)

function pwClassifier = estimateSigma(pwClassifier, features)
    
    if(nargin != 2)
        print_usage();
    elseif(!isa(pwClassifier, "parzenWindowClassifier") || !ismatrix(features))
        error("@parzenWindowClassifier/getStandardDeviation: requires \
parzenWindowClassifier, matrix");
    endif
    
	# expected value
	mu = sum(features, 1) ./ size(features, 1);
	
	# variance
	var = 0;
	for i = 1:size(features, 1)
		var += (features(i, :) .- mu) .^ 2;
	endfor
    
	pwClassifier.sigma = sqrt(var ./ (size(features, 1)-1));

endfunction