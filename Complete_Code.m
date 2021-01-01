%% Linear Regression, Gradient Descent and Cost Function

%By ArunKumar Ramachandran


%%In this code, the functions of both gradientDescent and computeCost are
%%called here and the program is run in order to depict the values of
%%gradient descent and anticipated cost function


% x refers to the population size in 10,000s
% y refers to the profit in $10,000s

%% Initialization
clear ; close all; clc

%% ======================= Part 1: Plotting =======================

% Loading dataset
data = load('ex1data1.txt');
X = data(:, 1)/10000; y = data(:, 2)/10000;
m = length(y); % number of training examples

% Plot Data
fprintf('Plotting Data ...\n')
plotData(X, y);

%% =================== Part 2: Cost and Gradient descent ===================


X = [ones(m, 1), data(:,1)/10000]; % This function returns ones for the hypothesis function
theta = zeros(2, 1); % Initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);


%% Result
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%% Prediction
% Predict values for population sizes of 35,000 and 70,000

predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% End of Program