# usage: acc = computeAccuracy(classifier, testFeat, testLab)

function acc = computeAccuracy(classifier, testFeat, testLab)
    
    acc = [];
    
    if(nargin != 3)
        print_usage();
    elseif(!isa(classifier, "classifier") || !ismatrix(testFeat) || !isvector(testLab))
        error("@classifier/computeAccuracy: requires classifier, matrix, vector");
    elseif(rows(testFeat) != length(testLab))
        error("@classifier/computeAccuracy: uneven number of feature vectors and labels");
    endif
    
    # classify each instance
    [classProb, label] = max(classifyInstances(classifier, testFeat));
    
    # acc = num of correctly classified instances / total instance number
    acc = sum((label' .- 1) == testLab) ./ length(testLab);
    
endfunction