# usage: labelNum = getNumberOfLabels(classifier)

function labelNum = getNumberOfLabels(classifier)

    labelNum = [];
    
    if(nargin != 1)
        print_usage();
    elseif(!isa(classifier, "classifier"))
        error("@classifier/getNumberOfLabels: requires classifier");
    endif
    
    labelNum = classifier.labelNum;

endfunction