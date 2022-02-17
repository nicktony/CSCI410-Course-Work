% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI410)
% % % % % % % % % % % % % % % % % % %


% Close command window, workspace, and all figure pop-ups
%--------------------------------------------------------------------
clc
clear all
close all
warning('off','all') % disable warning for matlabs version of k-kmeans

% Load in data
%--------------------------------------------------------------------
load fisheriris

% Assign data variables, additonal column for group type
%--------------------------------------------------------------------
Data = [meas(:,3:4) zeros(length(meas), 1)];

% Assigned clustering values by splitting data in thirds
k = 3; % 3 Initial Clusters
for i=1:length(Data)
    if (i <= length(Data)/k)
        Data(i,3:3) = 1;
    elseif (i <= length(Data)/k * 2)
        Data(i,3:3) = 2;
    elseif (i <= length(Data)) 
        Data(i,3:3) = 3;
    end
end


%--------------------------------------------------------------------
% PART ONE - K-Means Clustering From Book
%--------------------------------------------------------------------


% Calculate initial centroids
%--------------------------------------------------------------------
centroidGroupOneX = mean(Data(1:50, 1:1));
centroidGroupOneY = mean(Data(1:50, 2:2));
centroidGroupTwoX = mean(Data(51:100, 1:1));
centroidGroupTwoY = mean(Data(51:100, 2:2));
centroidGroupThreeX = mean(Data(101:150, 1:1));
centroidGroupThreeY = mean(Data(101:150, 2:2));

% Step through data and continously calculate centroids
%--------------------------------------------------------------------
recalcCentroids = 0;
i = 1;
while i <= length(Data)
    % Declare test variables
    X1 = Data(i,1:1);
    Y1 = Data(i,2:2);
    
    % Find distances from centroids
    euclideanDistance1 = sqrt((centroidGroupOneX - X1)^2 + (centroidGroupOneY - Y1)^2);
    euclideanDistance2 = sqrt((centroidGroupTwoX - X1)^2 + (centroidGroupTwoY - Y1)^2);
    euclideanDistance3 = sqrt((centroidGroupThreeX - X1)^2 + (centroidGroupThreeY - Y1)^2);

    % Calculate which class the test variable belongs to
    if (euclideanDistance1 < euclideanDistance2 ...
        && euclideanDistance1 < euclideanDistance3 ...
        && Data(i,3:3) ~= 1)
        
        % Set cluster grouping type
        Data(i,3:3) = 1;
        
        % Reasign centroid values
        recalcCentroids = 1; 

    elseif (euclideanDistance2 < euclideanDistance1 ...
            && euclideanDistance2 < euclideanDistance3 ...
            && Data(i,3:3) ~= 2)

        % Set cluster grouping type
        Data(i,3:3) = 2;
        
        % Reasign centroid values
        recalcCentroids = 1; 
        
    elseif (euclideanDistance3 < euclideanDistance1 ... 
            && euclideanDistance3 < euclideanDistance2 ...
            && Data(i,3:3) ~= 3)
        
        % Set cluster grouping type
        Data(i,3:3) = 3;
        
        % Reasign centroid values
        recalcCentroids = 1; 
        
    end
    
    % Recalculate centroid values if needed
    if (recalcCentroids == 1)
        idx = find(Data(:,3:3) == 1);
        centroidGroupOneX = mean(Data(idx,1:1));
        centroidGroupOneY = mean(Data(idx,2:2));
        
        idx = find(Data(:,3:3) == 2);
        centroidGroupTwoX = mean(Data(idx,1:1));
        centroidGroupTwoY = mean(Data(idx,2:2));
        
        idx = find(Data(:,3:3) == 3);
        centroidGroupThreeX = mean(Data(idx,1:1));
        centroidGroupThreeY = mean(Data(idx,2:2));
        
        % Reset i
        i = 0;
    end
    recalcCentroids = 0;
    i = i + 1;
end

% Display groupings of values
%--------------------------------------------------------------------
figure('NumberTitle', 'off', 'Name', 'Figure 1: Self-created K-means Clustering');

idx = find(Data(:,3:3) == 1);
scatter(Data(idx,1:1), Data(idx,2:2), 'filled');
hold on
idx = find(Data(:,3:3) == 2);
scatter(Data(idx,1:1), Data(idx,2:2), 'filled');
idx = find(Data(:,3:3) == 3);
scatter(Data(idx,1:1), Data(idx,2:2), 'filled');

title('Fisher''s Iris Data');
xlabel('Petal Lengths (cm)');
ylabel('Petal Widths (cm)');
legend('Group 1','Group 2','Group 3','Location','SouthEast');
hold off

% Display accuracy of program
%--------------------------------------------------------------------
accuracy = zeros(length(Data), 1);
for i=1:length(Data)
    if (Data(i,3:3) == 1 && strcmp(species{i}, 'setosa'))
        accuracy(i) = 1;
    elseif (Data(i,3:3) == 2 && strcmp(species{i}, 'versicolor'))
        accuracy(i) = 1;
    elseif (Data(i,3:3) == 3 && strcmp(species{i}, 'virginica'))
        accuracy(i) = 1;
    end
end

fprintf('%s\n','-----------------Self-Created K-Means-----------------');
fprintf('K-means Accuracy: %4.2f', (100 * sum(accuracy == 1)) / (length(Data)));
fprintf('%s\n\n', '%');


%--------------------------------------------------------------------
% PART TWO - K-Means Clustering From MATLAB
%--------------------------------------------------------------------


% Use the petal lengths and widths as predictors
X = meas(:,3:4);

% Cluster the data. Specify k = 3 clusters.
rng(1); % For reproducibility
[idx,C] = kmeans(X,3);

% Use kmeans to compute the distance from each centroid to points on a grid
x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid,3,'MaxIter',1,'Start',C);

% Assign each node in the grid to the closest centroid
figure('NumberTitle', 'off', 'Name', 'Figure 2: MATLAB K-means Clustering');
gscatter(XGrid(:,1),XGrid(:,2),idx2Region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(X(:,1),X(:,2),'k*','MarkerSize',5);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)'; 
legend('Region 1','Region 2','Region 3','Data','Location','SouthEast');
hold off;

% kmeans displays a warning stating that the algorithm did not converge, 
% which you should expect since the software only implemented one iteration.

% Report detailing my k-means vs. matlabs k-means
%--------------------------------------------------------------------
fprintf('%s\n','-----------------Report-----------------');
fprintf(['When comparing my scatter plot to the one created by MATLABs implementation, ', ...
'\nI found the groupings of each centroid to be identical. Both algorithms were \nable to predict ', ...  
'results accurately.']);
fprintf('\n\n');


%--------------------------------------------------------------------
% FUNCTIONS, END OF PROGRAM
%--------------------------------------------------------------------

          
% Save file contents
filename = 'A6_CSCI410_NickolausWhite.mat';
save(filename);

% Clear temporary variables
clearvars cv filename i idx k meas ...
          testOne testTwo trainOne trainTwo ...
          X1 X2 Y1 Y2 accuracy centroidGroupOneX ...
          centroidGroupOneY centroidGroupTwoX ...
          centroidGroupTwoY centroidGroupThreeX ...
          centroidGroupThreeY idx2Region recalcCentroids ...
          x1 x1G x2 x2G XGrid C euclideanDistance1 ...
          euclideanDistance2 euclideanDistance3
      
      
      
      