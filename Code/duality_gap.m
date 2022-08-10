function [gap] = duality_gap( patterns_train, labels_train, lambda, w, ell)
% Return the SVM duality gap
    
    n = numel(patterns_train); % number of examples
    ystars = cell(n,1);        % initialization of the predicted labels
    
    % get predicted label for each example
    for i=1:n
        ystars{i} = oracle(w, patterns_train{i}, labels_train{i});
    end
    
    w_s = zeros(size(w));
    ell_s = 0;

    % computation of necessary elements to compute the duality gap
    for i=1:n
        psi_i = featuremap(patterns_train{i}, labels_train{i})-featuremap(patterns_train{i}, ystars{i});
        w_s = w_s + psi_i/(lambda*n);
        ell_s = ell_s + loss(labels_train{i}, ystars{i})/n;
    end
      
    
    % computing duality gap:
    gap = lambda* w'*(w - w_s) - ell + ell_s;
    
end
