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

classOne1 = part1Data.classOne;
classTwo1 = part1Data.classTwo;

nBins = 100;

% Class distributions histogram
%--------------------------------------------------------------------
figure('NumberTitle', 'off', 'Name', 'Figure 1: Class Distributions');
histogram(classOne1, nBins, 'FaceColor',[0.2,0.2,0.5]);
hold on
histogram(classTwo1, nBins, 'FaceColor',[0,0.7,0.7]);
hold off
fontSize = 15;
xlabel('Class Values', 'FontSize', fontSize);
ylabel('Count', 'FontSize', fontSize);
legend('classOne', 'classTwo');

% Calculate prior probability of classOne and classTwo
%--------------------------------------------------------------------
priorP1 = round(length(classOne1)/(length(classOne1) + length(classTwo1)) * 100, 2);
priorP2 = round(length(classTwo1)/(length(classOne1) + length(classTwo1)) * 100, 2);

% Display prior probability's
%--------------------------------------------------------------------
fprintf('%s\n','-----------------Part 1-----------------');
fprintf('Prior probability of classOne: %4.2f', priorP1);
fprintf('%s\n','%');
fprintf('Prior probability of classTwo: %4.2f', priorP2);
fprintf('%s\n\n','%');

% Create random partitions of the two datasets
%--------------------------------------------------------------------
[trainOne1,~,testOne1] = dividerand(classOne1,0.6,0,0.4);
[trainTwo1,~,testTwo1] = dividerand(classTwo1,0.6,0,0.4);

% Calculate maximum likelihood estimates, mean and standard deviation
%--------------------------------------------------------------------
mleOfClassOne = mle(trainOne1, 'Distribution', 'Normal');
mean1 = mleOfClassOne(1:1);
stdDev1 = mleOfClassOne(2:2);

mleOfClassTwo = mle(trainTwo1, 'Distribution', 'Normal');
mean2 = mleOfClassTwo(1:1);
stdDev2 = mleOfClassTwo(2:2);

% Calculate prediction accuracy for claseOne & classTwo
%--------------------------------------------------------------------
sumOfClassOne1 = 0;
sumOfClassTwo1 = 0;
sumOfClassOne2 = 0;
sumOfClassTwo2 = 0;

% I sepereated these for-loops to eliminate any issues if the classes were
% different sizes (hypothetically)
for i = 1:length(testOne1)
    predAccOne1 = computeGaussianDensity(mean1,stdDev1,testOne1(i:i)); % test for class one
    predAccOne2 = computeGaussianDensity(mean2,stdDev2,testOne1(i:i)); % test for class two
    
    if (predAccOne1 > predAccOne2)
        sumOfClassOne1 = sumOfClassOne1 + 1;
    elseif (predAccOne2 > predAccOne1)
        sumOfClassTwo1 = sumOfClassTwo1 + 1;
    end
end

for i = 1:length(testTwo1)
    predAccTwo1 = computeGaussianDensity(mean1,stdDev1,testTwo1(i:i));
    predAccTwo2 = computeGaussianDensity(mean2,stdDev2,testTwo1(i:i));
    
    if (predAccTwo1 > predAccTwo2)
        sumOfClassOne2 = sumOfClassOne2 + 1;
    elseif (predAccTwo2 > predAccTwo1)
        sumOfClassTwo2 = sumOfClassTwo2 + 1;
    end
end

if (sumOfClassOne1 > sumOfClassTwo1)
    fprintf('%d', sumOfClassOne1);
    fprintf('%s',' > ');
    fprintf('%d', sumOfClassTwo1);

    predAccOne = round(sumOfClassOne1/length(testOne1) * 100, 2);
    fprintf(', thus, the first test data belongs to class one with a predication accuracy of %4.2f', predAccOne);
    fprintf('%s\n','%');
else
    fprintf('%d',sumOfClassOne2);
    fprintf('%s',' < ');
    fprintf('%d', sumOfClassTwo2);

    predAccOne = round(sumOfClassTwo1/length(testOne1) * 100, 2);
    fprintf(', thus, the first test data belongs to class two with a predication accuracy of %4.2f', predAccOne);
    fprintf('%s\n','%');
end

if (sumOfClassOne2 > sumOfClassTwo2)
    fprintf('%d', sumOfClassOne2);
    fprintf('%s',' > ');
    fprintf('%d', sumOfClassTwo2);
    
    predAccTwo = round(sumOfClassOne2/length(testOne2) * 100, 2);
    fprintf(', thus, the second test data belongs to class one with a predication accuracy of %4.2f', predAccTwo);
    fprintf('%s\n','%');
else
    fprintf('%d',sumOfClassOne2);
    fprintf('%s',' < ');
    fprintf('%d', sumOfClassTwo2);
    
    predAccTwo = round(sumOfClassTwo2/length(testTwo1) * 100, 2);
    fprintf(', thus, the second test data belongs to class two with a predication accuracy of %4.2f', predAccTwo);
    fprintf('%s\n','%');
end


%--------------------------------------------------------------------
% PART TWO
%--------------------------------------------------------------------


% Load in data (differs based on your file location)
%--------------------------------------------------------------------
part2Data = load('partTwoData.mat'); % load the file

classOne2 = part2Data.classOne;
classTwo2 = part2Data.classTwo;

% Calculate prior probability of classOne and classTwo
%--------------------------------------------------------------------
priorP1 = round(length(classOne2)/(length(classOne2) + length(classTwo2)) * 100, 2);
priorP2 = round(length(classTwo2)/(length(classOne2) + length(classTwo2)) * 100, 2);

% Display prior probability's
%--------------------------------------------------------------------
fprintf('%s\n\n','');
fprintf('%s\n','-----------------Part 2-----------------');
fprintf('Prior probability of classOne: %4.2f', priorP1);
fprintf('%s\n','%');
fprintf('Prior probability of classTwo: %4.2f', priorP2);
fprintf('%s\n\n','%');

% Create random partitions of the two datasets (Train: 60%, Test: 40%)
%--------------------------------------------------------------------
cv = cvpartition(size(classOne2,1),'HoldOut',0.4); 
idx = cv.test;

trainOne2 = classOne2(~idx,:);
testOne2  = classOne2(idx,:);

cv = cvpartition(size(classOne2,1),'HoldOut',0.4);
idx = cv.test;

trainTwo2 = classTwo2(~idx,:);
testTwo2  = classTwo2(idx,:);

% Combine testing data and transpose
test2 = [transpose(testOne2) transpose(testTwo2)];
test2 = transpose(test2);

% Calculate maximum likelihood estimates, mean and coveriance matrix
%--------------------------------------------------------------------
mean1 = mean(trainOne2, 1);
cov1 = cov(trainOne2);

mean2 = mean(trainTwo2, 1);
cov2 = cov(trainTwo2);

% Calculate prediction accuracy for claseOne & classTwo
%--------------------------------------------------------------------
r = size(testOne2, 1) + size(testTwo2, 1);

% Create matrices for results and accuracy calculations
results = zeros(1, r);
accuracy = results;

% Classify testing variables
for i = 1:length(test2)
    predAccOne1 = computeGaussianDensityMultivariate(mean1,cov1,test2(i,:));
    predAccOne2 = computeGaussianDensityMultivariate(mean2,cov2,test2(i,:));
    
    if (predAccOne1 > predAccOne2)
        results(i) = 1;
    elseif (predAccOne2 > predAccOne1)
        results(i) = 2;
    end
end

% Get accuracy for predictions
DataDIV = size(results, 2) - size(testOne2, 1);
for i = 1:size(results, 2)
    if i < DataDIV
        if results(i) == 1
            accuracy(i) = 1;
        end
    else
        if results(i) == 2
            accuracy(i) = 1;
        end
    end
end

% Calculate and output results
numCorrect = sum(accuracy == 1);
numIncorrect = sum(accuracy == 0);
totalTest = size(results, 2);

fprintf('Prediction Accuracy: %4.2f', (numCorrect/totalTest)* 100);
fprintf('%s\n','%');


%--------------------------------------------------------------------
% FUNCTIONS, END OF PROGRAM
%--------------------------------------------------------------------

% Save file contents
%---------------------------------------------------------------
filename = 'A1_CSCI410_NickolausWhite.mat';
save(filename);

% Bayesian classifier function
%--------------------------------------------------------------------
function z = computeGaussianDensity(m,S,x)
    [l,q] = size(m);  % l dimensionality
    z = (1/((2*pi)^l/2*det(S)^0.5)) * exp(-0.5 * (x-m)' * inv(S) * (x-m));
end

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




