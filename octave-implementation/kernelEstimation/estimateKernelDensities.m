# usage: densities = estimateKernelDensities(instances, samples, sigma)

function densities = estimateKernelDensities(instances, samples, sigma, useFix)

    densities = [];
    
	useEstimate = true;
	
    if(nargin >= 3)
        if(!ismatrix(instances) || !ismatrix(samples) || !isvector(sigma))
            error("kernelEstimation/estimateKernelDensities(3): requires matrix, \
matrix, vector");
        elseif(size(instances, 2) != size(samples, 2))
            error("kernelEstimation/estimateKernelDensities(3): instances and \
samples must have same number of columns");
        endif
        kernel = @(x) exp(-sum(x .* x, 2) ./ 2) ./ sqrt(2 * pi);
		if(nargin == 4)
			useEstimate = useFix;
		endif
    else
        print_usage();
    endif
	
	if(rows(samples) < 1)
		densities = zeros(1, rows(instances));
	else
		# compute standard deviation of samples
		# if not enough samples are available, take provided sigma
        # (std. dev. of whole dataset)
		estSigma = 0;
		if(size(samples, 1) > 1)
			mu = sum(samples, 1) ./ size(samples, 1);
			estSigma = sqrt(sum((samples .- mu) .^ 2, 1) ./ (size(samples, 1) - 1.5));
		endif
		
		if(useEstimate)
			if(estSigma != 0)
				sigma = estSigma;
			endif
			# estimate 'smoothness' based on Silverman's rule of thumb
			q = size(samples, 2);	# dimensions
			v = 2;					# kernel order (order of first non-zero moment)
			Cv = getSilvermanConstantFor2ndOrderGauss(q);
			bandwidth = bandwidth = sigma .* (Cv * size(samples, 1)^(-1/(2*v+q)));
		else
			bandwidth = sigma;
		endif
		
		# create matrices to match each instance with each sample
		sampleMat = repmat(samples, [rows(instances), 1]);
		instanceMat = reshape(repmat(instances', [rows(sampleMat)/rows(instances), 1]),
							 columns(sampleMat), rows(sampleMat))';
		
		# estimate the frequencies using the kernel provided (multivariate)
		densities = kernel((instanceMat .- sampleMat) ./ bandwidth);
		densities = reshape(densities, rows(samples), rows(instances));
		densities = max(sum(densities, 1) ./ (size(samples, 1) *...
                                                prod(bandwidth)), 10^(-99));
	endif
    
endfunction