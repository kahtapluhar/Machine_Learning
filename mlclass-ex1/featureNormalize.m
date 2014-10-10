function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly

X_norm = X; %copy of X
mu = mean(X) %[x y]
sigma = std(X); % [x y]
 

for i=1:size(X,2)
	X_norm(:,i)=X_norm(:,i)-mu(i);
	X_norm(:,i)=X_norm(:,i)/sigma(i);
end
    
end
