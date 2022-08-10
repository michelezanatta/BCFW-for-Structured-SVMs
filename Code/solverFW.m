function [w, gap_tt, avg_loss_tt, iter, time] = solverFW(patterns_train, labels_train, gap_threshold, time_budget, maxit)
% [model, progress] = solverBCFW(param, options)
%
% Solves the structured support vector machine (SVM) using the block-coordinate
% Frank-Wolfe algorithm.
% 
% Inputs:
%   
%
%     patterns_train
%         The tranining set.
%     
%     labels_train
%         The labels for the examples in the training set.
%      
%     gap_threshold
%         The threshold for the duality gap below which the algorithm
%         stops.
%
%     time_budget
%         Time limit for the algorithm.
% 
% Outputs:
%    
%     w 
%       The model parameters.
%
%     gap_tt
%       Vector containing the different duality gaps computed during the 
%       algorithm
%   
%     avg_loss_tt
%       Vector containing the loss on the training set 


% initialize the parameters

tic();
n = size(patterns_train, 2); % number of training examples
d = length(featuremap(patterns_train{1}, labels_train{1})); % dimension of feature mapping
lambda = 20; %regularization parameter
w = zeros(d, 1);
ell = 0;
gap_tt = 100;
avg_loss_tt = 1;
time_counter = 0;
time = 0;


fprintf("#############################\n");
fprintf("#############################\n");
fprintf('Running FW on %d examples\n', length(patterns_train));
fprintf("#############################\n");
fprintf("#############################\n");



% Main cycle
for k=0:maxit
    w_s = zeros(d,1);
    ell_s = 0;
    % cycle through all examples ("Batch" approach)
    for i = 1:n
        
        xi = patterns_train{i}; yi = labels_train{i};
        
        % call for the oracle to solve max problem
        ystar_i = oracle(w, xi, yi);
        
        % update quantities
        psi_i =   featuremap(xi, yi) - featuremap(xi, ystar_i);
        
     
        w_s = w_s + psi_i/(n*lambda);
        ell_s = ell_s + loss(yi, ystar_i)/n;
       
    end


    % step size
    gamma = 2/(2+k);

    
    
    % compute duality gap (this call is free here because all the needed
    % quantities are already computed)
    gap = lambda*(w'*(w - w_s)) - (ell - ell_s);

    gap_tt = [gap_tt; gap];
    error = average_loss(patterns_train, labels_train, w);
    avg_loss_tt = [avg_loss_tt; error];
    
    % stop if duality gap is below threshold:
    if gap <= gap_threshold
        fprintf('Duality gap below threshold -- stopping!\n')
        fprintf('current gap: %g, gap_threshold: %g\n', gap, gap_threshold)
        fprintf('Reached at iteration %d.\n', k+1)
        time_counter = time_counter + t_elapsed;
        time = [time; time_counter];
        break % exit loop!
    else
        fprintf('Duality gap check: gap = %g at iteration %d \n', gap, k+1)
    end

    %update final quantities
    w = (1-gamma)*w   + gamma*w_s;
    w = subplus(w);
    ell = (1-gamma)*ell + gamma*ell_s;
   
    
    % check for time budget
    t_elapsed = toc();
    time_counter = time_counter + t_elapsed;
    time = [time; time_counter];
    if (t_elapsed > time_budget)
        fprintf('time budget exceeded.\n');
        break
    end


end

gap_tt = gap_tt(2:end);
avg_loss_tt = avg_loss_tt(2:end);
time = time(2:end);
iter = 0:k;
end



