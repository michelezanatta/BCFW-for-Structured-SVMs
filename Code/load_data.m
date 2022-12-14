function [patterns_train, labels_train, patterns_test, labels_test] = load_data()     

    % loads training data
    load('coNLL_train.mat');
    num_words = size(X,1);
    Xtrain = [ones(num_words,1) X]; %saves features
    ytrain = y; %saves labels
    %saves start and end of sentence for division of dataset into samples
    sentencesTrain = initSentences_train(ytrain, length(ytrain));
    

    % loads test data
    load('coNLL_test.mat');
    num_words = size(X,1);
    Xtest = [ones(num_words,1) X];
    ytest = y;
    sentencesTest = initSentences_test(ytest, length(ytest));


    num_samples = size(sentencesTrain,1); % number of sentences for train
    num_samples_test = size(sentencesTest,1); % number of sentences for test
    num_features = max(Xtrain); %number of features | prende il massimo valore per ogni tag | i valori sono in ordine | ogni volta che ce n'è uno nuovo viene aggiunto un numero
    num_states = max(ytrain); %number of states
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

    % Saves features and lables in the cell format needed for solver function
    patterns_train = {}; % for training
    labels_train = {};
    patterns_test = {}; % for testing
    labels_test = {};
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
        patterns_train{i} = patterns;
        %labels per sentence
        labels_train{i} = ytrain(sentencesTrain(i,1):sentencesTrain(i,2),:);
    end

    %Test data
    for i=1:num_samples_test
        patterns = [];
        %features per sentence
        patterns.data = Xtest(sentencesTest(i,1):sentencesTest(i,2),:);
        patterns.num_states = num_states;
        patterns.num_features = num_features;
        patterns.featureStart = featureStart;
        patterns_test{i} = patterns;
        %labels per sentence
        labels_test{i} = ytest(sentencesTest(i,1):sentencesTest(i,2),:);
    end




    