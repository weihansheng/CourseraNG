function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    predictions=X*theta;
    tmp0=theta(1,1)-(alpha/m)*sum((predictions-y).*X(:,1));
    tmp1=theta(2,1)-(alpha/m)*sum((predictions-y).*X(:,2));
    theta(1,1)=tmp0;
    theta(2,1)=tmp1;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
plot(1:num_iters, J_history, 'rx'); % Plot the data 
ylabel('Cost'); % Set the y?axis label 
xlabel('iter'); % Set the x?axis label
end
