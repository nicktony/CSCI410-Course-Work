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


% Multi-layer perceptron with 3 layers: input layer, hidden layer, output layer (using a sigmoid function)
%--------------------------------------------------------------------

% Define target outputs
ANDtargetOutput = [0; 0; 0; 1];
ORtargetOutput = [0; 1; 1; 1];
NANDtargetOutput = [1; 1; 1; 0];
NORtargetOutput = [1; 0; 0; 0];
XORtargetOutput = [0; 1; 1; 0];

% Call to main function with each target output
fprintf('\n<strong>ANDtargetOutput</strong>');
main(ANDtargetOutput);
fprintf('\n<strong>ORtargetOutput</strong>');
main(ORtargetOutput);
fprintf('\n<strong>NANDtargetOutput</strong>');
main(NANDtargetOutput);
fprintf('\n<strong>NORtargetOutput</strong>');
main(NORtargetOutput);
fprintf('\n<strong>XORtargetOutput</strong>');
main(XORtargetOutput);

fprintf(['\nThe network has the hardest time training the XORtargetOutput. This is ', ...
'because a \nperceptron can only converge on linearly separable data. XOR target output ', ...
'data is\nnot linearly separable. Utilizing a multi-layer perceptron helps eliminate this\nissue ', ...
'but does not fix it unless the correct weights are given to the last number\nin the target output. ', ...
'In this example (e.g. 0,1,1,0) it is the value 0. Here we must\ncompute the AND and give it a negative ', ...
'weight (e.g. -2), subtracting out the difference\nif AND is TRUE; the OR will keep its weight of 1. ', ...
'Doing so, the network will compute\nthe XOR similar to the simplified equation OR-AND*2.\n\n']);


%--------------------------------------------------------------------
% FUNCTIONS, END OF PROGRAM
%--------------------------------------------------------------------


% Save file contents
%---------------------------------------------------------------
filename = 'week4Solution_NickolausWhite.mat';
save(filename);

% Main function, called for each targetOutput
function main(targetOutput)
    % Define the learning rate and total iterations
    learningRate = 0.5;
    totalIterations = 500;

    % Define the size of the input layer and the hidden layer
    inputLayerNumber = 2;
    hiddenLayerNumber = 2;

    % Define the input and hidden layer:
    inputLayer = zeros(inputLayerNumber, 1);
    hiddenLayer = zeros(hiddenLayerNumber, 1);

    % Define the output layer
    outputLayer = 0;

    % Randomly assign the weights to the input and hidden layer
    inputLayerWeights = rand((inputLayerNumber + 1) ,hiddenLayerNumber) - .5 ;
    hiddenLayerWeights = rand((hiddenLayerNumber + 1), 1) - .5;

    % Define the input data
    inputLayer = [0 0; 0 1; 1 0; 1 1];

    % Define the variable `m’ as the number of samples
    m = size(targetOutput, 1);

    % Add the bias to the input and hidden layer
    inputLayerWithBias = [ones(m,1) inputLayer];
    hiddenLayerWithBias = zeros(hiddenLayerNumber + 1, 1);

    % Loop that steps through each of the samples one at a time
    for iter=1:totalIterations
     for i = 1:m
        hiddenLayerActivation = inputLayerWithBias(i, :) * inputLayerWeights;
        hiddenLayer = sigmoid(hiddenLayerActivation);

        %Add the bias to the hiddenLayer
        hiddenLayerWithBias = [1, hiddenLayer];
        outputLayer = sigmoid(hiddenLayerWithBias * hiddenLayerWeights);

        %Calculate the error
        deltaOutput = targetOutput(i) - outputLayer;
        deltaHidden(1) = (deltaOutput * hiddenLayerWeights(1)) .* ((hiddenLayerWithBias(1) * (1.0 - hiddenLayerWithBias(1))));
        deltaHidden(2) = (deltaOutput * hiddenLayerWeights(2)) .* ((hiddenLayerWithBias(2) * (1.0 - hiddenLayerWithBias(2))));
        deltaHidden(3) = (deltaOutput * hiddenLayerWeights(3)) .* ((hiddenLayerWithBias(3) * (1.0 - hiddenLayerWithBias(3))));

        % Fixed Step Gradient Descent - Update the weights
        hiddenLayerWeights(1) = hiddenLayerWeights(1) + (learningRate * (deltaOutput * hiddenLayerWithBias(1)));
        hiddenLayerWeights(2) = hiddenLayerWeights(2) + (learningRate * (deltaOutput * hiddenLayerWithBias(2)));
        hiddenLayerWeights(3) = hiddenLayerWeights(3) + (learningRate * (deltaOutput * hiddenLayerWithBias(3)));

        % Update each weight according to the part that they played
        inputLayerWeights(1,1) = inputLayerWeights(1,1) + (learningRate * deltaHidden(2) * inputLayerWithBias(i, 1));
        inputLayerWeights(1,2) = inputLayerWeights(1,2) + (learningRate * deltaHidden(3) * inputLayerWithBias(i, 1));

        inputLayerWeights(2,1) = inputLayerWeights(2,1) + (learningRate * deltaHidden(2) * inputLayerWithBias(i, 2));
        inputLayerWeights(2,2) = inputLayerWeights(2,2) + (learningRate * deltaHidden(3) * inputLayerWithBias(i, 2));

        inputLayerWeights(3,1) = inputLayerWeights(3,1) + (learningRate * deltaHidden(2) * inputLayerWithBias(i, 3));
        inputLayerWeights(3,2) = inputLayerWeights(3,2) + (learningRate * deltaHidden(3) * inputLayerWithBias(i, 3));
     end
    end

    % Output summary of trained network
    outputSummary(inputLayerWithBias, inputLayerWeights, hiddenLayerWeights, targetOutput, totalIterations);
end

% Sigmoid function
function a = sigmoid(z)
    a = 1.0 ./ (1.0 + exp(-z));
end

% Cost function, this function will only work for NN with just one output (k = 1)
function [averageCost] = costFunction(inputLayerWithBias,inputLayerWeights, hiddenLayerWeights, targetOutput)
    % Sum of square errors cost function
    m = 4;
    hiddenLayer = sigmoid(inputLayerWithBias * inputLayerWeights);
    hiddenLayerWithBias = [ones(m,1) hiddenLayer];
    outputLayer = sigmoid(hiddenLayerWithBias * hiddenLayerWeights);
    
    % Step through all of the samples and calculate the cost at each one
    for i=1:m
     cost(i) = (1/2) * ((outputLayer(i) - targetOutput(i)) .^ 2);
    end
    
    % Sum up all of the individual costs
    totalCost = sum(cost);
    
    % Average them out
    averageCost = totalCost * (1/m);
end

% Function that summarizes the output for the 4 samples
function outputSummary(inputLayerWithBias, inputLayerWeights, hiddenLayerWeights, targetOutput, totalIterations)
    cost = costFunction(inputLayerWithBias, inputLayerWeights, hiddenLayerWeights, targetOutput);
    hiddenLayer = sigmoid(inputLayerWithBias * inputLayerWeights);

    % We have multiple samples, add the bias to each of them
    hiddenLayerWithBias = [ones(size(targetOutput,1),1) hiddenLayer];
    actualOutput = sigmoid(hiddenLayerWithBias * hiddenLayerWeights);
    fprintf('\n=========================================\n');
    fprintf('Output Summary (after %d iterations):\n', totalIterations);
    fprintf('Total Cost: [%f]\n', cost);
    for i=1:length(actualOutput)
     if(actualOutput(i) > 0.5)
        thresholdedValue = 1;
     else
        thresholdedValue = 0;
     end

     if(thresholdedValue == targetOutput(i))
        fprintf('Sample[%d]: Target = [%f] Thresholded Value = [%f] Actual= [%f]\n', i, targetOutput(i), thresholdedValue, actualOutput(i));
     else % Else print the error in red
        fprintf(2,'Sample[%d]: Target = [%f] Thresholded Value = [%f] Actual= [%f]\n', i, targetOutput(i), thresholdedValue, actualOutput(i));
     end
    end
        fprintf('=========================================\n\n');
end




