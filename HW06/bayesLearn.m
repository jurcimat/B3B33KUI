function [ bayes ] = bayesLearn( train,train_labels )
%bayesLearn naive bayes classifier learning
%   [ bayes ] = bayesLearn( train,train_labels )
% train - matrix with training examples in rows
% train_labels - column with labels of the training examples (char array)
% bayes - structure with the learned classifier

%          TODO Here you have to implement your function for training naive bayes
%          classifier

num_of_features = size(train,2);
num_of_examples = size(train,1);
conversion_table = unique(train_labels);
num_of_classes = numel(conversion_table);
MAX_INTENSITY = 256;
K = 0.5;     % Laplace smoothing parameter was empirically discovered
% convert labels from train labels to indexes in conversion table
conversion_vector = zeros(num_of_examples);
for x = 1:numel(train_labels)
    conversion_vector(x) = find(conversion_table == train_labels(x));
end
% Create frequency table where is counted number of different intensities
% for each class feature
frequency_table = zeros(num_of_classes, num_of_features, MAX_INTENSITY);
for example = 1:num_of_examples
    for feature_index = 1:num_of_features
        % increment by one because of indexing in MATLAB
        index_of_intensity = train(example,feature_index) + 1;
        class = conversion_vector(example);
        frequency_table(class, feature_index, index_of_intensity) = ...
            frequency_table(class, feature_index, index_of_intensity) + 1;
    end
end
        
% Convert frequency table to probability table
probability_table = zeros(num_of_classes, num_of_features, MAX_INTENSITY);
for feature = 1:num_of_features
    for intensity = 1:MAX_INTENSITY
        count = 0;
        % Get total count feature with given intensity in all classes
        for class = 1:num_of_classes
            count = count + frequency_table(class,feature,intensity);
        end
        % Convert to probability using Laplace smoothing
        for class = 1:num_of_classes
            probability_table(class,feature,intensity) = ...
                (frequency_table(class, feature, intensity) + K)/...
                    (count + num_of_features*K);
        end
    end
end

bayes.num_of_features = num_of_features;
bayes.num_of_classes = num_of_classes;
bayes.probability_table = probability_table;
bayes.conversion_table = conversion_table;


end % end of function

