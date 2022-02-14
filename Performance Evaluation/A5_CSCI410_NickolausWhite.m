% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI410)
% % % % % % % % % % % % % % % % % % %


% Close command window, workspace, and all figure pop-ups
%--------------------------------------------------------------------
clc
clear all
close all

% Load in data
%--------------------------------------------------------------------
load fisheriris

% Split data into two classes, using the first 100 samples
%--------------------------------------------------------------------
classOne = meas(1:50,3:4);
classTwo = meas(51:100,3:4);


%--------------------------------------------------------------------
% PART ONE - Bayesian Classifier
%--------------------------------------------------------------------


% Create random partitions of the two datasets (Train: 60%, Test: 40%)
%--------------------------------------------------------------------
cv = cvpartition(length(classOne),'HoldOut',0.4); 
idx = cv.test;

trainOne = classOne(~idx,:);
testOne  = classOne(idx,:);

cv = cvpartition(length(classTwo),'HoldOut',0.4);
idx = cv.test;

trainTwo = classTwo(~idx,:);
testTwo  = classTwo(idx,:);

% Combine testing data and transpose
testData = [transpose(testOne) transpose(testTwo)];
testData = transpose(testData);

% Calculate maximum likelihood estimates, mean and coveriance matrix
%--------------------------------------------------------------------
mean1 = mean(trainOne, 1);
cov1 = cov(trainOne);

mean2 = mean(trainTwo, 1);
cov2 = cov(trainTwo);

% Calculate prediction accuracy for claseOne & classTwo
%--------------------------------------------------------------------
r = length(testOne) + length(testTwo);

% Create matrices for results and accuracy calculations
predictBayesian = zeros(1, r);
accuracy = predictBayesian;

% Classify testing variables
for i = 1:length(testData)
    pred1 = computeGaussianDensityMultivariate(mean1,cov1,testData(i,:));
    pred2 = computeGaussianDensityMultivariate(mean2,cov2,testData(i,:));
    
    if (pred1 > pred2)
        predictBayesian(i) = 1;
    elseif (pred2 > pred1)
        predictBayesian(i) = 2;
    else
        predictBayesian(i) = NaN;
    end
end

% Get accuracy for predictions
dataDiv = length(predictBayesian) - length(testOne);
for i = 1:length(predictBayesian)
    if i < dataDiv
        if (predictBayesian(i) == 1)
            accuracy(i) = 1;
        end
    else
        if (predictBayesian(i) == 2)
            accuracy(i) = 1;
        end
    end
end

% Calculate and output results
numCorrectBayesian = sum(accuracy == 1);
numIncorrectBayesian = sum(accuracy == 0);

fprintf('%s\n','-----------------Part 1-----------------');
fprintf('Bayesian Classifier Prediction Accuracy: %4.2f', (numCorrectBayesian/length(predictBayesian))* 100);
fprintf('%s\n\n','%');


%--------------------------------------------------------------------
% PART TWO - K-nearest Neighbor
%--------------------------------------------------------------------


% Step through class one test samples (Nearest Neighbor)
%--------------------------------------------------------------------
k = 1; % Set k equal to 1
for i=1:20
    % Declare test variables
    X1 = testOne(i,1:1);
    Y1 = testTwo(i,2:2);
    
    % Declare training variables, use the Distance Formula to calculate distances
    for j=1:30
        X2 = trainOne(j,1:1);
        Y2 = trainOne(j,2:2);
        euclideanDistanceClassOne(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
        
        X2 = trainTwo(j,1:1);
        Y2 = trainTwo(j,2:2);
        euclideanDistanceClassTwo(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
    end

    % Calculate which class the test variable belongs to
    if (sum(euclideanDistanceClassOne <= k) > sum(euclideanDistanceClassTwo <= k))
        predictKnn1(i) = 1;
    else
        predictKnn1(i) = 2;
    end
end
numCorrectKnnClassOne = sum(predictKnn1 == 1);
numIncorrectKnnClassOne = sum(predictKnn1 == 2);

% Step through class two test samples (Nearest Neighbor)
%--------------------------------------------------------------------
k = 1; % Set k equal to 1
for i=1:20
    % Declare test variables
    X1 = testTwo(i,1:1);
    Y1 = testTwo(i,2:2);
    
    % Declare training variables, use the Distance Formula to calculate distances
    for j=1:30
        X2 = trainOne(j,1:1);
        Y2 = trainOne(j,2:2);
        euclideanDistanceClassOne(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
        
        X2 = trainTwo(j,1:1);
        Y2 = trainTwo(j,2:2);
        euclideanDistanceClassTwo(j) = sqrt((X2 - X1)^2 + (Y2 - Y1)^2);
    end
    
    % Calculate which class the test variable belongs to
    if (sum(euclideanDistanceClassOne <= k) > sum(euclideanDistanceClassTwo <= k))
        predictKnn2(i) = 1;
    else
        predictKnn2(i) = 2;
    end
end
numCorrectKnnClassTwo = sum(predictKnn2 == 2);
numIncorrectKnnClassTwo = sum(predictKnn2 == 1);

% Combine prediction values for Knn for confusion matrix
predictKnn = cat(2, predictKnn1, predictKnn2);    

% Print confusion matrices
%--------------------------------------------------------------------
fprintf('%s\n','-----------------Part 2-----------------');
fprintf('K-Nearest Neighbor Prediction Accuracy: %4.2f', 100 * (numCorrectKnnClassOne + numCorrectKnnClassTwo) / (2 * (20)));
fprintf('%s\n\n', '%');

% Create species array for confusion matrix
species = zeros(1, 40);
for i=1:40
    if (i <= 20)
        species(i) = 1;
    else
        species(i) = 2;
    end
end

figure('NumberTitle', 'off', 'Name', 'Figure 1: Bayesian Classifier');
confusionchart(species, predictBayesian);

figure('NumberTitle', 'off', 'Name', 'Figure 2: K-Nearest Neighbor');
confusionchart(species, predictKnn);


%--------------------------------------------------------------------
% FUNCTIONS, END OF PROGRAM
%--------------------------------------------------------------------

          
% Save file contents
filename = 'A5_CSCI410_NickolausWhite.mat';
save(filename);

% Clear temporary variables
clearvars accuracy cov1 cov2 cv dataDiv ...
          euclideanDistanceClassOne euclideanDistanceClassOne ...
          filename i idx j k mean1 mean2 meas ...
          pred1 pred2 predictBayesian predictKnn ...
          predictKnn1 predictKnn2 r X1 X2 Y1 Y2

% Bayesian classifier multivariate function
%--------------------------------------------------------------------
function z = computeGaussianDensityMultivariate(m,C,x)
    [l,q] = size(m); %1 dimensionality
    
    % Broken down for matrices to work together
    e = ((2*pi)^l/2*det(C)^0.5);
    z_scor = (x-m)';
    g = -0.5 * z_scor;
    h = g *(x-m)*inv(C);
    z = 1/e * exp(h);
end




