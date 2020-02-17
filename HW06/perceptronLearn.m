function [ perceptron ] = perceptronLearn( train, train_labels )
%perceptronLearn perceptron algorithm 
%   [ perceptron ] = perceptronLearn( train, train_labels )
% train - matrix with training examples in rows
% train_labels - column with labels of the training examples (char array)
% perceptron - structure with the learned classifier

%create conversion table (char labels --> integer labels)
conversionTable=unique(train_labels);

%number of classes
numClasses=numel(conversionTable);

% number of features
numFeatures=size(train,2);

%number of training examples
numExamples=size(train,1);

% convert chars from train_labels to integers and save them to variable trainLabelsConv 
trainLabelsConv=zeros(numExamples,1);
for i=1:numel(train_labels)
trainLabelsConv(i)=find(conversionTable==train_labels(i));
end

% setting default values
W=zeros(numFeatures,numClasses);
B=zeros(1,numClasses);

% Repeat training until error-free classification is found
 while true
% calculate activations of the perceptrons
prod=train*W+repmat(B,numExamples,1);  

% find the perceptron with maximal activation
[~, class]=max(prod,[],2);

% find some misclassified example
misclass=find((class~=trainLabelsConv),1);

% error free??
if isempty(misclass)
break;
end 

% Increase weights in correct perceptron
corrClass=trainLabelsConv(misclass);
W(:,corrClass)=W(:,corrClass)+train(misclass,:)';
B(corrClass)=B(corrClass)+1;

% Decrease weights in incorect perceptron
incorrClass=class(misclass);
W(:,incorrClass)=W(:,incorrClass)-train(misclass,:)';
B(incorrClass)=B(incorrClass)-1;

 end

%  save to the perceptron structure
 perceptron.W=W;
 perceptron.B=B;
 perceptron.conversionTable=conversionTable;


end

