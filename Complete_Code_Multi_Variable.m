%% Machine Learning Linear regression with multiple variables

% By Arunkumar Ramachandran 


%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1_data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% fprintf('Program paused. Press enter to continue.\n');
% pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================


%
%               First task is to  make sure that the functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that we should try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, we should run the complete code at the end
%               to predict the price of a 1650 sq-ft area for the restaurant.
%
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

x = zeros(1,2);
x(1) = input("\n Enter size in sq ft: ");
x(2) = input("\n Enter number of dinning rooms: ");
% x(3) = input("\n Enter a crime rate: ");

x = (x - mu)./sigma;
x = [1 x]';

% Estimate the price of the sq-ft  for the restaurant
price = theta'*x; % You should change this


fprintf(['Predicted price of the sq-ft for the restaurant ' ...
         '(using gradient descent):\n $%f\n'], price);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ================ Part 3: Normal Equations ================

%%fprintf('Solving with normal equations...\n');


% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. The code in 
%               normalEqn.m should be completed.
%
%               After doing so, we should run the complete code 
%               to predict the price of a 1650 sq-ft area for the
%               restaurant.
%

%% Load Data
% data = csvread('ex1_data2.txt');
% X = data(:, 1:3);
% y = data(:, 4);
% m = length(y);
% % Add intercept term to X
% X = [ones(m, 1) X];
% 
% % Calculate the parameters from the normal equation
% theta = normalEqn(X, y);
% 
% % Display normal equation's result
% fprintf('Theta computed from the normal equations: \n');
% fprintf(' %f \n', theta);
% fprintf('\n');
% 
% 
% % Estimate the price of a 1650 sq-f area for the restaurant
% price = 0; 
% 
% 
% fprintf(['Predicted price of a 1650 sq-ft area for the restaurant' ...
%          '(using normal equations):\n $%f\n'], price);
% 
