function [nodePot,edgePot] = makeLogNodePotentials(xi,w,v_start,v_end,v)
% Make Log-Potentials for given sentence
% solita inizializzazione di parametri
num_states = xi.num_states;
num_features = xi.num_features;
featureStart = xi.featureStart;
num_featuresTotal = featureStart(end)-1;
num_nodes = size(xi.data,1);

% Make node potentials

nodePot = zeros(num_nodes,num_states); % Matrice parole X stati
for n = 1:num_nodes % Per ogni parola
    features = xi.data(n,:); % features for word w in given sentence
    for state = 1:num_states % per ogni stato
        pot = 0; % inizializzo pot
        for f = 1:length(num_features) % per feature
            if features(f) ~= 0 % we ignore features that are 0
                featureParam = featureStart(f)+features(f)-1; 
                pot = pot+w(featureParam+num_featuresTotal*(state-1));
            end
        end
        nodePot(n,state) = pot;
    end
end
nodePot(1,:) = nodePot(1,:) + v_start'; % Modification for beginning of sentence
nodePot(end,:) = nodePot(end,:) + v_end'; % Modification for end of sentence

% Transitions are not dependent on features, so are position independent
edgePot = exp(v); 

