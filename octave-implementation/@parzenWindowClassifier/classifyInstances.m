# usage: ret = classifyInstances(pwc, instances)

function ret = classifyInstances(pwc, instances, useFreq)

    ret = [];
    
    if((nargin != 2) && (nargin != 3))
        print_usage();
    elseif(!isa(pwc, "parzenWindowClassifier") || !ismatrix(instances))
        error("@parzenWindowClassifier/classifyInstances: requires \
parzenWindowClassifier, matrix");
    elseif(sum(size(instances)) < 2)
        error("@parzenWindowClassifier/classifyInstances: instance dimensions \
must be at least 1x1");
	elseif(nargin != 3)
		useFreq = false;
    endif
    
    [features, ~, labelInd] = getTrainingInstances(pwc);
    
    densities = zeros(length(labelInd), size(instances, 1));
    priorEstimates = zeros(length(labelInd), 1);
	
	kernel = @(x) exp(-sum(x .^ 2, 2) ./ (2*0.1^2));
    
    old = 1;
    # for each class label
    for i = 1:length(labelInd)
        priorEstimates(i) = (labelInd(i) - old + 1) / size(features, 1);
        
        # check if training instances for the class are present
        if(labelInd(i) - old >= 0)
            # use kernel density estimation (Parzen-Window)
			if(useFreq)
				densities(i, :) = estimateKernelFrequencies(instances,
									features(old:labelInd(i), :), kernel);
			else
				densities(i, :) = estimateKernelDensities(instances,
									features(old:labelInd(i), :), 0.1, false);
			endif
        endif
        
        old = labelInd(i)+1;
    endfor
    
    # use density estimation and estimated prior probabilities to estimate posteriors
    ret = (priorEstimates .* densities) ./ sum(priorEstimates .* densities, 1);
    
endfunction