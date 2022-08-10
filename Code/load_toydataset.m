function [patterns_train_toy, labels_train_toy, patterns_test_toy, labels_test_toy] = load_toydataset(n, m)     
    %Function to load a toy dataset. The dataset is still the coNLL but the
    %number of words is lower. The parameters n,m define the number of
    %sentences for the train and test dataset.
    
    % loads training data
    load('coNLL_train.mat');
    %saves labels
    ytrain = y;
    %saves start and end of sentence for division of dataset into samples
    %for only n sentences. num_words is the total number of words for n
    %sentences
    [sentencesTrain, num_words] = initSentences_train(ytrain, n);
    ytrain = ytrain(1:num_words);
    X = X(1:num_words, :);
    Xtrain = [ones(num_words,1) X]; %saves features
    
 

    % loads test data
    load('coNLL_test.mat');
    ytest = y;
    [sentencesTest, num_words] = initSentences_test(ytest, m);
    ytest = ytest(1:num_words);
    X = X(1:num_words, :);
    Xtest = [ones(num_words,1) X];

    num_samples = size(sentencesTrain,1); % number of sentences for train | se tutto giusto, dovrebbe essere == n. Tengo scritto così perché in load_data utilizziamo questo modo
    num_samples_test = size(sentencesTest,1); % number of sentences for test | se tutto giusto, dovrebbe essere == m
    num_features = max(Xtrain); %number of features | prende il massimo valore per ogni tag | i valori sono in ordine | ogni volta che ce n'è uno nuovo viene aggiunto un numero
    num_states = max(ytrain); %number of states | preso sul train set
    featureStart = cumsum([1 num_features(1:end)]); % numero di features in totale

    clear X y sentences % You better not be accessing these instead {Xtrain,ytrain}
    
    % From double to int32
    Xtrain = int32(Xtrain);
    ytrain = int32(ytrain);
    sentencesTrain = int32(sentencesTrain);
    Xtest = int32(Xtest);
    ytest = int32(ytest);
    sentencesTest = int32(sentencesTest);
    featureStart = int32(featureStart);
    num_words = size(ytrain,1);
    num_states = int32(num_states);

    % NOTA: il numero degli stati varia in base al numero di frasi scelto.
    % Se le frasi sono prese in ordine, il numero degli stati è ordinato. 
    % Per fare una verifica veloce, verificare con unique(ytrain).

    % Saves features and lables in the cell format needed for solver function
    patterns_train_toy = {}; % for training
    labels_train_toy = {};
    patterns_test_toy = {}; % for testing
    labels_test_toy = {};
    idx_train = 1;
    idx_test = 1;

    %Training data
    for i=1:num_samples
        patterns = [];
        %features per sentence
        patterns.data = Xtrain(sentencesTrain(i,1):sentencesTrain(i,2),:); % prende una singola frase
        patterns.num_states = num_states;                             
        patterns.num_features = num_features;
        patterns.featureStart = featureStart;
        patterns_train_toy{i} = patterns;
        %labels per sentence
        labels_train_toy{i} = ytrain(sentencesTrain(i,1):sentencesTrain(i,2),:);
    end

    %Test data
    for i=1:num_samples_test
        patterns = [];
        %features per sentence
        patterns.data = Xtest(sentencesTest(i,1):sentencesTest(i,2),:);
        patterns.num_states = num_states;
        patterns.num_features = num_features;
        patterns.featureStart = featureStart;
        patterns_test_toy{i} = patterns;
        %labels per sentence
        labels_test_toy{i} = ytest(sentencesTest(i,1):sentencesTest(i,2),:);
    end