function i = randsample_fast(weights)
% randsample_fast is a speeded up version of MATLAB's randsample.m built-in function.
% computes the gap_sampling

n = numel(weights);
if n == 0
    error([mfilename, ': empty weight vector']);
end

sum_weights = sum(weights);
if ~(sum_weights > 0) || ~all(weights>=0) 
    error([mfilename, ': weights are not good']);
end


probs = weights(:)' / sum_weights;

edges = min([0 cumsum(probs)],1); % if sum of probs gets bigger than one
                                  % because of round off we fix it

edges(n+1) = 1; % get the upper edge exact
[~, i] = histc(rand, edges);

end