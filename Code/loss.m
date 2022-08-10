function loss = loss(ytruth, ypredict)
% Returns the normalized Hamming distance of predicted label ypredict to true

loss = sum(ypredict~=ytruth) / numel(ytruth);