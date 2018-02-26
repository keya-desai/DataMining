

clear ; close all; clc
X=  csvread('diabetes.csv',1,0,[1 0 768 7]);
y = csvread('diabetes.csv',1,8);


m = length(y)

%Normalizing Features

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Choose some alpha value
alpha = 0.01;
num_iters = 10000;

theta = ones(9, 1);

[theta, J_history] = gradientDescentMulti1(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
 fprintf(' %f \n', theta);
 fprintf('\n');
 
 fprintf(['Cost function value is %f \n'], J_history(end));
   

