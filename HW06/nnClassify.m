function classLabels=nnClassify(nn,test)
%nnClassify 1-nearest neighbourn classification
%   classLabels=nnClassify(nn,test)
% nn - structure with the learned nn (see nnLearn)
% test - matrix with testing examples in rows
% classLabels - column with labels of the classified examples (char array)

%          TODO Here you have to implement classification function for 1-nearest
%          neighbor


num_of_test_features = size(test,2);
num_of_items_for_classification = size(test,1);
% test whether number of features are equal
if nn.num_of_features ~= num_of_test_features
    disp("ERROR: Wrong number of features between training data and tested data");
    return
end
% Preallocation of classLabels to increase speed of algorithm
classLabels = repmat('c',num_of_items_for_classification,1);

% BRUTE FORCE method for 1-nearest neighbour
for test_index = 1:num_of_items_for_classification
    min_error = [Inf, '']; % initialize error difference
    
    for index_in_train_data = 1:nn.num_of_examples
        % Calculate difference between tested data and trainnig data
        % It is sum of differences between feature vectors of tested and training data
        error = sum(abs(test(test_index,:)-nn.training_data(index_in_train_data,:)));
        
        if error < min_error(1)
            % Change if new error is less than previous stored error
            min_error(1) = error;
            min_error(2) = nn.training_data_class_labels(index_in_train_data);
        end
    end
    % Classify as the feature vector from training data with the smallest 
    % difference from tested data
    classLabels(test_index,1) = min_error(2);
    
end

end % end of function

