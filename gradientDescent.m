%%
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
 J=(X*theta)-y;
 
 temp0=theta(1)-((alpha/m)*sum(J.*X(:,1)));
 temp1=theta(2)-((alpha/m)*sum(J.*X(:,2)));
 
 theta=[temp0;temp1];

 % Save the cost J in every iteration    
 J_history(iter) = computeCost(X, y, theta);

end
end