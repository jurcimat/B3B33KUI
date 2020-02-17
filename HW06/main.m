% Programming exercise – alphanumeric characters recognition

clear all

%loading training data
load('train.mat')
load('train_labels.mat')

%loading test data
load('test.mat')
load('test_labels.mat')



% % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%example solution - perceptron algorithm
disp('Linear classifier learned by perceptron algorithm:')

% learning
% perceptron=perceptronLearn(train,train_labels);

% classification
% classLabelsPerc=perceptronClassify(perceptron,test);

% print confusion matrix and show result
% confusionMatrix(classLabelsPerc,test_labels);



% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%1-nearest neighbour classifier
% implement functions nnLearn.m and nnClassify.m
disp('1-nearest neighbour classifier:')

% learning
nn=nnLearn(train,train_labels);

% classification
classLabelsNn=nnClassify(nn,test);

% print confusion matrix and show result
confusionMatrix(classLabelsNn,test_labels);



% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%naive Bayes classifier
% implement functions bayesLearn.m and bayesClassify.m
disp('1-naive Bayes classifier:')

% learning
bayes=bayesLearn(train,train_labels);

% classification
classLabelsBayes=bayesClassify(bayes,test);

% print confusion matrix and show result
confusionMatrix(classLabelsBayes,test_labels);


