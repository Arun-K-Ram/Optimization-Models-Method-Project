function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear 

%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

%theta=0;
% You need to return the following variables correctly 
%Hypothesis is given as:
m = length(y);
h = X*theta;
J=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost
J=(1/(2*m))*sum((h-y).^2);
% =========================================================================

end
