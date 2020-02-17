function classLabels=bayesClassify(bayes,test)
%bayesClassify naive bayes classification
%   classLabels=bayesClassify(bayes,test)
% bayes - structure with the learned classifier (see bayesLearn)
% test - matrix with testing examples in rows
% classLabels - column with labels of the classified examples (char array)

%          TODO Here you have to implement classification function for naive bayes
%          classifier

num_of_test_features = size(test,2);
num_of_items_for_classification = size(test,1);
% test whether number of features are equal
if bayes.num_of_features ~= num_of_test_features
    disp("ERROR: Wrong number of features between training data and tested data");
    return
end
% Preallocation of classLabels to increase speed of algorithm
classLabels = repmat('c',num_of_items_for_classification,1);

for test_index = 1:num_of_items_for_classification
    max_probability = {-Inf,''}; % initialize max probability for specified class

    for class = 1:bayes.num_of_classes
        % Calculate difference between tested data and trainnig data
        % It is sum of differences between feature vectors of tested and training data
        probability = 0;
        for feature = 1:bayes.num_of_features
            % count probability using logarithm theorem from lectures
            probability = probability + ...
            log(bayes.probability_table(class,feature, test(test_index,feature) + 1));
        end

        if probability > max_probability{1}
            % Change if current probability is bigger than last stored
            max_probability{1} = probability;
            max_probability{2} = bayes.conversion_table(class);
        end
    end
    % Classify as the class with highest probability
    classLabels(test_index,1) = max_probability{2};    
end


end

