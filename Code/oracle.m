function label = oracle(w, xi, yi)
% Do loss-augmented decoding on a given example (xi,yi) using
% model.w as parameter. Param is ignored (included for standard
% interface). The loss used is normalized Hamming loss.
% 
% If yi is not given, then standard prediction is done (i.e. MAP decoding
% without the loss term).

if issparse(w)
    w = full(w);
end

num_states = xi.num_states;
num_features = xi.num_features;
featureStart = xi.featureStart;
num_featuresTotal = featureStart(end)-1;
data = xi.data;


v_start = w(num_featuresTotal*num_states+1:num_featuresTotal*num_states+num_states); % prende vettore relativo a g_start | stesse coordinate
v_end = w(num_featuresTotal*num_states+num_states+1:num_featuresTotal*num_states+2*num_states); % prender vettore relativo a g_end

v = reshape(w(num_featuresTotal*num_states+2*num_states+1:end),num_states,num_states); % matrice relativa a bin | stesse coordinate

wlog = reshape(w(1:num_featuresTotal*num_states),num_featuresTotal,num_states); % matrice relativa a unit

num_Nodes = size(xi.data,1); % Ogni parola è un nodo
    
% Make Potentials
logNodePot = makeLogNodePotentials(xi,wlog,v_start,v_end,v); 

% Add loss-augmentation to the score (normalized Hamming distance used for loss)
% Make Loss-Augmented Potentials
if nargin > 2
    logNodePot = logNodePot + 1/num_Nodes;
    for n = 1:num_Nodes
        logNodePot(n,yi(n)) = logNodePot(n,yi(n)) - 1/num_Nodes;
    end
end
% Solve inference problem
yMAP = logDecode(logNodePot,v);
label = yMAP;

end