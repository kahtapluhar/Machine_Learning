function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    %theta 0 Equation
    derivate0 = (1 / m) * sum((X * theta) - y);
    theta0 = theta(1, 1) - (alpha * derivate0);
    
    %theta 1 Equation
    derivate1 = (1 / m) * sum(((X * theta) - y) .* X(:, 2));
    theta1 = theta(2, 1) - (alpha * derivate1);
    
    %Return Values
    theta = [theta0; theta1];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
