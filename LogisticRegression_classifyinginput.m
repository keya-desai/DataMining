
% code where 50% of dataset is used to train the logistic
% regressor and other 50 % to test it

clear ; close all; clc
X=  csvread('diabetes.csv',1,0,[1 0 768 7]);
y = csvread('diabetes.csv',1,8);



X_train= X(1:384,:)
y_train= y(1:384,:)



m = length(y_train)

%Normalizing Features

[X_train mu sigma] = featureNormalize(X_train);

% Add intercept term to X
X_train = [ones(m, 1) X_train];

% Choose some alpha value
alpha = 0.01;
num_iters = 20000;

theta = ones(9, 1);



[theta, J_history] = gradientDescentMulti1(X_train, y_train, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
 fprintf(' %f \n', theta);
 fprintf('\n');
 
 fprintf(['Cost function value is %f \n'], J_history(end));
   

% Predictind the output

x_test= X(385:767,:);
y_test= y(385:767,:)
[x_test mu sigma] = featureNormalize(x_test);
x_test = [ones(383, 1) x_test];

hypothesis = 1 ./ (1+ exp(-(x_test * theta)) );


for k=1:383
    if( hypothesis(k,1) < 0.5 )
     hypothesis(k,1) = 0;
    else 
     hypothesis(k,1) = 1;
    end 

end

%disp(hypothesis)

C= confusionmat(y_test, hypothesis);

disp('Confusion Matrix:')
disp(C)

accuracy = ((C(1,1)+ C(2,2)) ./ (C(1,1)+ C(2,2)+ C(1,2)+ C(2,1)) ) * 100 ;

disp('TP:');
disp(C(1,1));
disp('TN:');
disp(C(2,2));
disp('FP:');
disp(C(1,2));
disp('FN:');
disp(C(2,1));

disp('Accuracy(in %):')
disp(accuracy);

%% Additional Test cases

a=  [ 5 , 153, 76, 40, 120, 36.1, 0.471,26 ; [ 4 , 90 , 70, 30 , 0, 35.7 , 0.27,53 ]; [ 4 , 90 , 70, 30 , 0, 35.7 , 0.27,53 ]] ;
[a mu sigma] = featureNormalize(a);
a = [ones(3, 1) a];

hypothesis = 1 ./ (1+ exp(-(a * theta)) );
for k=1:3
    if( hypothesis(k,1) < 0.5 )
     hypothesis(k,1) = 0;
    else 
     hypothesis(k,1) = 1;
    end 

end
disp(hypothesis)


