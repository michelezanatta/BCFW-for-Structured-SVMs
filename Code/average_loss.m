function [ avg_loss ] = average_loss( patterns, labels, w)
% Return the average loss for the predictions of model.w on
% input data param.patterns. See solverBCFW for interface of param.
%
% This function is expensive, as it requires a full decoding pass over all
% examples (so it costs as much as n BCFW iterations).

    avg_loss = 0;
    for i=1:numel(patterns)
        ystar_i = oracle(w, patterns{i}); % standard decoding (not loss-augmented) as no input label , labels{i}
        avg_loss = avg_loss + loss(labels{i}, ystar_i);
    end
    avg_loss = avg_loss / numel(patterns);
    
end % average_loss
