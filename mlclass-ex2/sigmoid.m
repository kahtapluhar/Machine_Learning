function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

for i=1:size(z,1)
	for j=1:size(z,2)
		g(i,j)=1/(1+exp(-z(i,j)));
	end;
end;

end
