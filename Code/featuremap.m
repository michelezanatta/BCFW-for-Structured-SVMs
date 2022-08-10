function phi = featuremap(xi, yi)
% FEATUREMAP 
% xi is a sentence, composed of a number of words. 
% yi is the label for each word.

num_states = xi.num_states; % number of possible state for each word of xi

featureStart = xi.featureStart; % vector with cumsum of possible params for each word of xi
num_featuresTotal = featureStart(end)-1; % number of total features, binary representation for sparsity | -1 perchè la prima feature è di solo 1 (vedere linea 32 load_toydataset)
data = xi.data; % words with tags for sentence xi

unit = zeros(num_featuresTotal,num_states); % create a matrix features x states of zeros
gr_start = zeros(num_states,1);
gr_end = zeros(num_states,1);
bin = zeros(num_states);

% Note that xi.data is given already by its unary features. In the 
% following, these are transformed into a binary representation and biases
% for the beginning and the end of the sentence as well as sequential 
% features are added.
num_features = xi.num_features; % Vettore con colonne tipo di feature e valori numero di possibili valori per quella feature
nNodes = size(xi.data,1); % numero di parole nella frase
% Update gradient
for n = 1:nNodes % unary features   | per ogni parola
    features = xi.data(n,:);    % rappresentazione della parola
    for feat = 1:length(num_features) % per ogni feature
        if features(feat) ~= 0 % 
            featureParam = featureStart(feat)+features(feat)-1; % crea variabile che prende il numero di variabili utilizzate per le features precedenti, somma il valore della feature della parola corrente e toglie 1. Per trovare indice da utilizzare nel vettore unit.
            for state = 1:num_states % per state nei diversi stati
                O = (state == yi(n)); % output se lo stato è quello corretto della parola
                unit(featureParam,state) = unit(featureParam,state) + O; % update unit: featureParam è l'indice del parametro relativo ad una feature, state è la colonna dello stato, e mette output precedente.
            end
         end
     end
end

% Ad ora abbiamo analizzato una frase. Ci siamo creati una matrice (unit) di
% dimensioni il numero di parametri possibili unendo tutte
% le features per il numero degli states possibili. Poi per ogni parola
% abbiamo analizzato il valore di ogni feature e modificato la matrice in
% questo modo:
% Prendiamo lo stato della parola - relativa colonna della matrice - e
% aggiungiamo 1 ai parametri della parola. Quindi se la parola ha
% rappresentazione il vettore [1 5 8 10] e stato 4, modifichiamo la quarta
% colonna della matrice aggiungendo 1 alle righe {1 5 8 10}.
% In altre parole, abbiamo una rappresentazione 74'000 x 22 per ogni frase sul dataset totale.
% featureParam x states

for state = 1:num_states        % per ogni stato
    O = (state == yi(1));       % se lo stato corrisponde allo stato della prima parola
    gr_start(state) = gr_start(state) + O; % beginning of sentence | questo valore cambia solo se lo stato corrisponde allo stato della prima parola
    O = (state == yi(end));     % se lo stato corrisponde allo stato dell'ultima parola
    gr_end(state) = gr_end(state) + O; % end of sentence | questo valore cambia solo se lo stato corrisponde allo stato dell'ultima parola
end

% gr_start e gr_end hanno le coordinate = 1 solo negli stati delle parole
% ad inizio e fine frase.

% Adesso guardiamo gli stati delle parole successive alla parola
% selezionata
for n = 1:nNodes-1 % sequential binary features | per ogni parola tranne l'ultima
    for state1 = 1:num_states % per ogni stato - relativo alla parola di riferimento
        for state2 = 1:num_states % per ogni stato - relativo alla parola successiva
            O = ((state1 == yi(n)) && (state2 == yi(n+1))); % trova combinazione di stati. O = 1 solo se x_{t} ha stato s1 e x_{t+1}
            bin(state1,state2) = bin(state1,state2) + O; % update bin: matrice n_states x n_states
        end
    end
end

% Bin è la transition map


% Quindi adesso abbiamo, per ogni frase, una matrice bin num_states x
% num_states che per ogni stato (riga) ci dice quante volte lo stato
% (colonna) lo ha seguito nella frase. Esempio:
% Frase: Il clima è molto bello 
% Stati: 1    1   2   2     2
% Bin matrice [1 1; 0 2]

% Arriviamo alla feature map vera e propria: phi crea una matrice di
% dimensioni (74'659 + 1 + 1 + 22) x 22 unendo le varie matrici e vettori
% precedenti (molto sparsa quindi), e la trasforma utilizzando la funzione
% sparse(A)

% Per vedere un esempio di come funziona la funzione sparse, provare
% seguente:
% A = [0 0 0 0 0 0 0 1 0 0; 0 0 0 0 1 0 0 1 0 0; 0 0 0 0 0 1 0 0 0 0; 1 0 0 0 0 0 0 0 0 0; 2 0 0 0 0 0 0 0 1 0];
% sparse(A)

phi = [unit(:); gr_start;gr_end;bin(:)];
% transform into space feature
phi = sparse(phi);

end


