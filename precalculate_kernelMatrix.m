clc
clear
load('datasets/yeast-train.mat');
[n,d] = size(trainData);
L = size(trainLabel,2);

para_matrix = 2.^(-3:1:-1);

tic
for p = 1: size(para_matrix, 2)  %pre-compute the kernel matrix: km(i,j) = \phi(xi)*\phi(xj)
    scale = para_matrix(p);
    [kernelMatrix,~] = rbfkernel_call(trainData, scale);
    save(['kernel_matrix/yeast/kernelMatrix_scale_' num2str(scale) '.mat'],'kernelMatrix');
    disp(['success!  scale =' num2str(scale)]);
    clear kernelMatrix
end
toc
