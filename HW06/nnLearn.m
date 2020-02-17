function [ nn ] = nnLearn( train,trainLabels )
%nnLearn 1-nearest neighbour learning
%   [ nn ] = nnLearn( train,trainLabels )
% train - matrix with training examples in rows
% trainLabels - column with labels of the training examples (char array)
% nn - structure with the learned classifier

%         TODO Here you have to implement your function for training 1-nearest
%         neighbor classifier

% Get number of features from training data
nn.num_of_features = size(train,2);
% Get number of examples in trainng data
nn.num_of_examples = size(train,1);
% Store trainig data in classificator
nn.training_data = train;
% Store class labels of training data in classificator
nn.training_data_class_labels = trainLabels;



end

