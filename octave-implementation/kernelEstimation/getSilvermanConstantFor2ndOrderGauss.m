# usage: Cv = getSilvermanConstantFor2ndOrderGauss(dims)

function Cv = getSilvermanConstantFor2ndOrderGauss(dims)

	Cv = [];
	
	if(nargin != 1)
		print_usage();
	elseif(!isscalar(dims) || (floor(dims) != dims) || (dims < 1))
		error("kernelEstimation/getSilvermanConstantFor2ndOrderGauss: requires \
scalar integer >= 1");
	endif
	
	# precomputed constants array
	CvPre = [1.05922, 1.00000, 0.96863, 0.95058, 0.93971, 0.93303, 0.92893,...
			0.92648, 0.92514, 0.92453, 0.92443, 0.92469, 0.92520, 0.92587,...
			0.92667, 0.92755, 0.92849, 0.92946, 0.93044, 0.93143];
	
    
	if(dims > 20)
		# if > 20, we have to compute them on the fly
        # (unlikely in our case)
		Cv = getSilvermanConstant(@(x) exp(-x .^ 2 ./ 2) ./ sqrt(2*pi), dims);
	else
		Cv = CvPre(dims);
	endif

endfunction