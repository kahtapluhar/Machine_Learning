function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples
h=0;
    for i = 1:m
        h = h + (theta' *  X(i,:)' - y(i))^2; % Hypothesis EQ with theta0 = 0;
    end
J = (1/(2*m))*h; %Cost Equation
end


