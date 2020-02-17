function confusionMatrix(classLabels,testLabels)
%confusionMatrix print confusion matrix and the result of the classifier
%   confusionMatrix(classLabels,testLabels)
% classLabels - labels obtained by classification
% testLabels - given labels
% for more info see http://en.wikipedia.org/wiki/Confusion_matrix

if numel(classLabels)~=numel(testLabels)
error('size of classLabels and testLabels must be same');    
end

%number of examples
numExamples=numel(classLabels);

%create conversion table (char labels --> integer labels)
conversionTable=unique([classLabels;testLabels]);

% convert chars from classLabels and testLabels to integers and save them
classLabelsConv=zeros(numExamples,1);
testLabelsConv=zeros(numExamples,1);
for i=1:numExamples
classLabelsConv(i)=find(conversionTable==classLabels(i));
testLabelsConv(i)=find(conversionTable==testLabels(i));
end

% init the matrix
Confusion=zeros(numel(conversionTable));

% fill confusion matrix
for i=1:numExamples
Confusion(testLabelsConv(i),classLabelsConv(i))=Confusion(testLabelsConv(i),classLabelsConv(i))+1;
end

% print confusion matrix
array2table(Confusion,'VariableNames',cellstr(conversionTable),'RowNames',cellstr(conversionTable))

% trace(Confusion) - sum of diagonal elements
disp(['Results: ',num2str(trace(Confusion)),' of ',num2str(numExamples),' correctly classified.'])

fprintf('----------------\n')


end

