function [ classLabels ] = perceptronClassify( perceptron,test )
%perceptronClassify perceptron classification
%   [ classLabels ] = perceptronClassify( perceptron,test )
% perceptron - structure with the learned perceptron (see perceptronLearn)
% test - matrix with testing examples in rows
% classLabels - column with labels of the classified examples (char array)

% get perceptron parameters from the structure perceptron
W=perceptron.W;
B=perceptron.B;
conversionTable=perceptron.conversionTable;
 
 %number of testing examples
numExamples=size(test,1);
 
 % calculate activations of the perceptrons
prod=test*W+repmat(B,numExamples,1); 

% determine classes
[~, class]=max(prod,[],2);

% initialize classLabels
classLabels=blanks(numel(class))';

% convert integer labels to char labels and save them to classLabels
for i=1:numel(class)
classLabels(i)=conversionTable(class(i));
end


end

