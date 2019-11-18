function r = acc(label, result)
% Clustering Accuracy

    n = length(label);
    difference = abs(label - result);
    idx = find(difference ~= 0);
    r = 1 - length(idx)/n;
    
end