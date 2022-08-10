function [sentences, i] = initSentences_test(y, n)
% Get indices of starts of sentences. A sentence start when index == 0.
nWords = length(y);
sentences = zeros(0,2);
j = 1;
for i = 1:length(y)
    if (i==1 || y(i-1) == 0) && y(i) ~= 0
        sentences(j,1) = i;
    end
    if (i==nWords || y(i+1) == 0) && y(i) ~= 0
        sentences(j,2) = i;
        j = j + 1;
    end

    % break dopo n parole.
    if j == n+1
        break
    end
end

