

function [theta, J_history] = gradientDescentMulti1(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    hypothesis = 1 ./ (1 + exp(- (X * theta)) );
    %disp(hypothesis);
    error = hypothesis - y;
    newDecrement = ((1/m) * alpha * X' * error);
    theta = theta - newDecrement;
    
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti1(X, y, theta);

end

end
