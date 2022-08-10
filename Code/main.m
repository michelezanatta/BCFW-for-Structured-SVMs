%% Feature map su una sola frase

% Load a fixed number of sentences from the training set and, if needed
% from the test set.
n_sentences_train = 1000; n_sentences_test = 10;
[patterns_train, labels_train, patterns_test, labels_test] = load_toydataset(n_sentences_train, n_sentences_test);
 
% Exit conditions for the FW algorithm and sampling
gap_threshold = 0.01;
time_budget = 1000;
gap_check = 100;
maxit_block = 10000;
maxit_batch = 1000;

% Call to the FW algorithm "gap" sampling
sampling = "gap";
[w, gap_tt_gap, avg_loss_tt_gap, iter_gap, time_gap] = solverBCFW(patterns_train, labels_train, ...
    gap_threshold, time_budget, gap_check, sampling, maxit_block);

% Call to the FW algorithm "uniform" sampling
sampling = "uniform";
[w, gap_tt_uni, avg_loss_tt_uni, iter_uni, time_uni] = solverBCFW(patterns_train, labels_train, ...
    gap_threshold, time_budget, gap_check, sampling, maxit_block);


% Call to the batch FW algorithm 
[w, gap_tt_batch, avg_loss_tt_batch, iter_batch, time_batch] = solverFW(patterns_train, labels_train, ...
    gap_threshold, time_budget, maxit_batch);

plot_fw(iter_batch, iter_gap, iter_uni, time_batch, time_gap, time_uni, gap_tt_batch, gap_tt_gap, ...
    gap_tt_uni, avg_loss_tt_batch, avg_loss_tt_gap, avg_loss_tt_uni)

clc;