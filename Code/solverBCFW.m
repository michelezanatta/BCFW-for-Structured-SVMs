function [w,gap_tt, avg_loss_tt, iter, time] = solverBCFW(patterns_train, labels_train, gap_threshold, time_budget, gap_check, sampling, maxit)
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
%     gap_check
%         Since computing the duality gap is not free like in the batch
%         algorithm and needs to be computed on the whole set, we decided
%         to compute it every gap_check iterations.
%     
%     sampling
%         The type of sampling to be performed
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

n = length(patterns_train); % number of training examples

% general initializations

lambda = 20; % regularization parameter
phi1 = featuremap(patterns_train{1}, labels_train{1}); % use first example to determine dimension
d = length(phi1); % dimension of feature mapping
m = patterns_train{1}.num_states; % number of states


% inizialization of the parameters
w = zeros(d,1);
wMat = zeros(d,n); 
avg_loss_tt = 1;
gap_tt = [];
ell = 0; 
ellMat = zeros(n,1);
gap = 100;
gap_tt = [gap];
w_s = zeros(size(w));
ell_s=0;

fprintf("#############################\n");
fprintf("#############################\n");
fprintf('Running BCFW on %d examples\n', length(patterns_train));
fprintf("#############################\n");
fprintf("#############################\n");

% set a seed for to reproduce experiment
rand_seed = 10;
rand('state',rand_seed);
randn('state',rand_seed);
time_counter = 0;
time = [];
tic();


% Inizialization for the gap sampling
gap_vec = Inf(n, 1);


% Main loop 
gap_check_counter = 1; % keeps track of how many passes through the data since last duality gap check

for k=1:maxit
    % pick a random example according to the chosen sampling method
    if sampling=="uniform"
        i = randi(n);
    else
        i = randsample_fast(gap_vec);
    end

    gap_vec(gap_vec<0) = 0;

        
    % call for the oracle to solve the max problem
    ystar_i = oracle(w, patterns_train{i}, labels_train{i});
            
    % update the quantities
    psi_i =   featuremap(patterns_train{i}, labels_train{i}) - featuremap(patterns_train{i}, ystar_i);
    w_s = 1/(n*lambda) * psi_i;
    loss_i = loss(labels_train{i}, ystar_i);
    ell_s = 1/n*loss_i;
    
    % update the vector for the gap sampling
    gap_i_FW = lambda*(w'*(wMat(:,i) - w_s)) - (ellMat(i) - ell_s);
    gap_vec(i) = max(gap_i_FW,0);
   
    % get the step size
    gamma = 2*n/(k+2*n);
            
    % update weights and ell
    w = w - wMat(:,i); % this is w^(k)-w_i^(k) on the paper
    wMat(:,i) = (1-gamma)*wMat(:,i) + gamma*w_s;
    w = w + wMat(:,i); % this is w^(k+1) = w^(k)-w_i^(k)+w_i^(k+1) on the paper
    
    ell = ell - ellMat(i); % this is ell^(k)-ell_i^(k) on the papaer
    ellMat(i) = (1-gamma)*ellMat(i) + gamma*ell_s;
    ell = ell + ellMat(i); % this is ell^(k+1) = ell^(k)-ell_i^(k)+ell_i^(k+1) on the paper
    
  
    % check for time budget
    t_elapsed = toc();
    if (t_elapsed > time_budget)
        fprintf('Time budget exceeded.\n');
        break;
    end
    
    % checking duality gap stopping criterion if required:
    if gap_check_counter >= gap_check
        gap_check_counter = 0;
                      
        % compute gap:
        gap = duality_gap(patterns_train, labels_train, lambda, w, ell);
        if gap <= gap_threshold
            fprintf('Duality gap below threshold -- stopping!\n')
            fprintf('current gap: %g, gap_threshold: %g\n', gap, gap_threshold)
            fprintf('Reached at iteration %d.\n', k)
             gap_tt = [gap_tt; gap];
            % compute average loss
            error = average_loss(patterns_train, labels_train, w);
            avg_loss_tt = [avg_loss_tt; error];
            time_counter = time_counter + toc();
            time = [time; time_counter];
            break % exit loop
        else
            fprintf('Duality gap check: gap = %g at iteration %d\n', gap, k)
        end
        
        gap_tt = [gap_tt; gap];
        % compute average loss
        error = average_loss(patterns_train, labels_train, w);
        avg_loss_tt = [avg_loss_tt; error];
        time_counter = time_counter + toc();
        time = [time; time_counter];
    end 
    
    gap_check_counter = gap_check_counter+1;
    

end 
time = [0; time];
iter = (0:(k/gap_check))*gap_check;
end 
