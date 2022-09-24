% A fast way to compute rbf kernel : k(x,y)=exp(-||x-y||^2/2*scale)
function [K,d] = rbfkernel_call(data,scale)

X = data';
Y = data';
K = X'*Y;
[~,Mx] = size(X);
[~,My] = size(Y);

K = 2*K;
K = K - sum(X.^2,1)'*ones(1,My);
K = K - ones(Mx,1)*sum(Y.^2,1);
d=-K;
K = exp(K/(2*scale));




