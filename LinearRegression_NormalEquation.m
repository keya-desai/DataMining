X = csvread('boston_housing.csv', 1,0, [1 0 506 12]);
Y = csvread('boston_housing.csv', 1,13, [1 13 506 13]);
Xt = X.';
M = inv(Xt*X);
W = M*Xt*y;
 
H = X*W - Y;
Hsq = H.^2;
sum = sum(Hsq);
 
cost = sum/(2*506);
 
a = [0.085, 13.0, 10.5, 1.0, 0.8, 4.78, 39.0, 5.5, 5.5, 331.0, 13.3, 390.5, 17.71];
b = a*W;
