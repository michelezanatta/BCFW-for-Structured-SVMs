function [w, ell, gap] = solverFW_duality_gap_fun(patterns_train, labels_train, maxit, gap_threshold, time_budget)
% [model, gaps, cache] = onePass_FW(param, options, model, cache)
%
% onePass_FW performs a batch FW step.
% Designed to be called from solver_BCFW_hybrid.m


%% get problem description
tic();
n = size(patterns_train, 2); % number of training examples
d = length(featuremap(patterns_train{1}, labels_train{1})); % dimension of feature mapping
% using_sparse_features = isfield(param, 'using_sparse_features') && param.using_sparse_features;
lambda = 50;
w = zeros(d, 1);
ell = 0;
ellMat = zeros(n, 1);


%% Initialize models and losses corresponding to the corner atom 
w_s = zeros(d,1);
wsMat = zeros(d,n);

ell_s = 0;
ell_s_mat = zeros(n,1);


%% through data
for k = 0:(maxit-1)
    ystars = cell(n,1);
    for i=1:n
        % solve the loss-augmented inference for point i
        ystars{i} = oracle(w, patterns_train{i}, labels_train{i});
    end
    w_s = zeros(size(w));
    ell_s = 0; %
    for i=1:n
        psi_i = featuremap(patterns_train{i}, labels_train{i})-featuremap(patterns_train{i}, ystars{i});
        w_s = w_s + psi_i/(n*lambda);
        ell_s = ell_s + loss(labels_train{i}, ystars{i})/n;
    end
 
    
   
    % step size
    gamma = 2/(2+k);

    
    
    % compute duality gap:
    gap = lambda*(w'*(w - w_s)) - (ell - ell_s);

    
   

    % stop if duality gap is below threshold:
    if gap <= gap_threshold
        fprintf('Duality gap below threshold -- stopping!\n')
        fprintf('current gap: %g, gap_threshold: %g\n', gap, gap_threshold)
        fprintf('Reached at iteration %d.\n', k+1)
        break % exit loop!
    else
        fprintf('Duality gap check: gap = %g at iteration %d \n', gap, k+1)
    end

    w = (1-gamma)*w   + gamma*w_s;
    w = subplus(w);
    ell = (1-gamma)*ell + gamma*ell_s;
   
    
    % time-budget exceeded?
    t_elapsed = toc();
    if (t_elapsed > time_budget)
        fprintf('time budget exceeded.\n');
        return
    end

end

