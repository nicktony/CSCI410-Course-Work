% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI410)
% % % % % % % % % % % % % % % % % % %


% Close command window, workspace, and all figure pop-ups
%--------------------------------------------------------------------
clc
clear all
close all


%--------------------------------------------------------------------
% PART ONE
%--------------------------------------------------------------------


% Load in data (differs based on your file location)
%--------------------------------------------------------------------
part1Data = load('partOneData.mat'); % load the file

classOne = part1Data.classOne;
classTwo = part1Data.classTwo;

% Split data into training and testing
%--------------------------------------------------------------------
numClassOneSamples = size(classOne, 2);
numClassTwoSamples = size(classTwo, 2);

% Class one
randomClassOne = randsample(numClassOneSamples, numClassOneSamples);
classOneTrainIndices = randomClassOne(1:6000);
classOneTestIndices = randomClassOne(6001:10000);

classOneTrainData = classOne(classOneTrainIndices);
classOneTestData = classOne(classOneTestIndices);

% Class two
randomClassTwo = randsample(numClassTwoSamples, numClassTwoSamples);
classTwoTrainIndices = randomClassTwo(1:6000);
classTwoTestIndices = randomClassTwo(6001:10000);

classTwoTrainData = classTwo(classTwoTrainIndices);
classTwoTestData = classTwo(classTwoTestIndices);

numTrainSamples = 6000;
numTestSamples = 4000;

% Step through class one test samples (Nearest Neighbor)
%--------------------------------------------------------------------
for i=1:numTestSamples
    for j=1:numTrainSamples
        euclideanDistanceClassOne(j) = sqrt((classOneTestData(i) - classOneTrainData(j))^2);
        euclideanDistanceClassTwo(j) = sqrt((classOneTestData(i) - classTwoTrainData(j))^2);
    end
    
    if (min(euclideanDistanceClassOne) < min(euclideanDistanceClassTwo))
        predict(i) = 1;
    else
        predict(i) = 2;
    end
end
correctClassOne = sum(predict == 1);
incorrectClassOne = sum(predict == 2);

% Step through class two test samples (Nearest Neighbor)
%--------------------------------------------------------------------
for i=1:numTestSamples
    for j=1:numTrainSamples
        euclideanDistanceClassOne(j) = sqrt((classTwoTestData(i) - classOneTrainData(j))^2);
        euclideanDistanceClassTwo(j) = sqrt((classTwoTestData(i) - classTwoTrainData(j))^2);
    end
    
    if (min(euclideanDistanceClassOne) < min(euclideanDistanceClassTwo))
        predict(i) = 1;
    else
        predict(i) = 2;
    end
end
correctClassTwo = sum(predict == 2);
incorrectClassTwo = sum(predict == 1);

% Print results
%--------------------------------------------------------------------
fprintf('%s\n','-----------------Part 1-----------------');

fprintf('Class one correct predictions: %d\n', correctClassOne);
fprintf('Class one correct predictions: %d\n\n', incorrectClassOne);

fprintf('Class two correct predictions: %d\n', correctClassTwo);
fprintf('Class two incorrect predictions: %d\n\n', incorrectClassTwo);

fprintf('Total correct predictions: %d\n', correctClassOne + correctClassTwo);
fprintf('Total incorrect predictions: %d\n', incorrectClassOne + incorrectClassTwo);
fprintf('Total prediction accuracy: %4.2f', 100 * (correctClassOne + correctClassTwo) / (2 * (numTestSamples)));
fprintf('%s\n\n', '%');


%--------------------------------------------------------------------
% PART TWO
%--------------------------------------------------------------------


% Load in data (differs based on your file location)
%--------------------------------------------------------------------
part2Data = load('partTwoData.mat'); % load the file

classOne = part2Data.classOne;
classTwo = part2Data.classTwo;

% Create random partitions of the two datasets (Train: 60%, Test: 40%)
%--------------------------------------------------------------------
cv = cvpartition(size(classOne,1),'HoldOut',0.4); 
idx = cv.test;

classOneTrainData = classOne(~idx,:);
classOneTestData  = classOne(idx,:);

cv = cvpartition(size(classTwo,1),'HoldOut',0.4);
idx = cv.test;

classTwoTrainData = classTwo(~idx,:);
classTwoTestData  = classTwo(idx,:);

numTrainSamples = 6000;
numTestSamples = 4000;

% Step through class one test samples (Nearest Neighbor)
%--------------------------------------------------------------------
k = 1; % Set k equal to 1
for i=1:numTestSamples
    % Declare test variables
    X1 = classOneTestData(i,1:1);
    Y1 = classOneTestData(i,2:2);
    
    % Declare training variables, use the Distance Formula to calculate distances
    for j=1:numTrainSamples
        X2 = classOneTrainData(j,1:1);
        Y2 = classOneTrainData(j,2:2);
        euclideanDistanceClassOne(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
        
        X2 = classTwoTrainData(j,1:1);
        Y2 = classTwoTrainData(j,2:2);
        euclideanDistanceClassTwo(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
    end

    % Calculate which class the test variable belongs to
    if (sum(euclideanDistanceClassOne <= k) > sum(euclideanDistanceClassTwo <= k))
        predict(i) = 1;
    else
        predict(i) = 2;
    end
end
correctClassOne = sum(predict == 1);
incorrectClassOne = sum(predict == 2);

% Step through class two test samples (Nearest Neighbor)
%--------------------------------------------------------------------
k = 1; % Set k equal to 1
for i=1:numTestSamples
    % Declare test variables
    X1 = classTwoTestData(i,1:1);
    Y1 = classTwoTestData(i,2:2);
    
    % Declare training variables, use the Distance Formula to calculate distances
    for j=1:numTrainSamples
        X2 = classOneTrainData(j,1:1);
        Y2 = classOneTrainData(j,2:2);
        euclideanDistanceClassOne(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
        
        X2 = classTwoTrainData(j,1:1);
        Y2 = classTwoTrainData(j,2:2);
        euclideanDistanceClassTwo(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
    end
    
    % Calculate which class the test variable belongs to
    if (sum(euclideanDistanceClassOne <= k) > sum(euclideanDistanceClassTwo <= k))
        predict(i) = 1;
    else
        predict(i) = 2;
    end
end
correctClassTwo = sum(predict == 2);
incorrectClassTwo = sum(predict == 1);

% Print results
%--------------------------------------------------------------------
fprintf('%s\n','-----------------Part 2-----------------');

fprintf('Class one correct predictions: %d\n', correctClassOne);
fprintf('Class one correct predictions: %d\n\n', incorrectClassOne);

fprintf('Class two correct predictions: %d\n', correctClassTwo);
fprintf('Class two incorrect predictions: %d\n\n', incorrectClassTwo);

fprintf('Total correct predictions: %d\n', correctClassOne + correctClassTwo);
fprintf('Total incorrect predictions: %d\n', incorrectClassOne + incorrectClassTwo);
fprintf('Total prediction accuracy: %4.2f', 100 * (correctClassOne + correctClassTwo) / (2 * (numTestSamples)));
fprintf('%s\n\n', '%');


%--------------------------------------------------------------------
% END OF PROGRAM
%--------------------------------------------------------------------


% Save file contents
%---------------------------------------------------------------
filename = 'A2_CSCI410_NickolausWhite.mat';
save(filename);




