% used for datasets with full, not sparse, features and labels

clc
clear
global trainData testData testLabel
dname = 'yeast';
load(['datasets/' dname '-train.mat']);
load(['datasets/' dname '-test.mat']);

epoch = 1;
times = 20;  % run 20 times for calculating mean metric values

[n,d] = size(trainData);
L = size(trainLabel,2);

%hyperparameters for kernelized FFLT
scale = 2.^(-2);   %RBF kernel hyperparameter
eta = 2.^(-1);
maxIterNum = 14;
gamma = 0; % fixed threshold


sr = RandStream.create('mt19937ar','Seed',1);
RandStream.setGlobalStream(sr);

%%for storing the performance metrics' values
test_macro_F1_score = zeros(times,1);
test_micro_F1_score = zeros(times,1);
hammingLoss = zeros(times,1);
rankingLoss = zeros(times,1);
subsetAccuracy = zeros(times,1);
oneError = zeros(times,1);
precision = zeros(times,1);
recall = zeros(times,1);
F1score = zeros(times,1);
testTime = zeros(times,1);

kernelMatrix = importdata(['kernel_matrix/' dname '/kernelMatrix_scale_' num2str(scale) '.mat']);

tStart = tic;
for run = 1:times
    coff = zeros(n, L);
    SVsIdx = zeros(1, n);
    SVsNum = 0;
    for o = 1:epoch
        index = randperm(n);
        for i=1:n
            j = index(i);
            x = trainData(j,:)';
            y = trainLabel(j,:);
            t = (o - 1)*n + i;
            
            if t == 1
                pred_v = zeros(1,L);
                km = [];
            else
                km = kernelMatrix(SVsIdx(1:SVsNum),j);
                pred_v = km'* coff(1:SVsNum,:);
            end
            
            % pred_y = pred_v(1:L) > gamma; %online prediction
            Y_t_size = nnz(y);
            
            for iter = 1:maxIterNum
                a_t = 0;
                b_t = 0;
                cur_coeff = zeros(1, L);
                for k = 1:L
                    if y(k) == 1 && (pred_v(k) - gamma < 1)
                        a_t = a_t + 1;
                        cur_coeff(k) = eta / Y_t_size;
                    elseif y(k) == 0 && (gamma -  pred_v(k) < 1)
                        b_t = b_t + 1;
                        cur_coeff(k) = - eta / (L - Y_t_size);
                    end
                end
                if a_t == 0 && b_t == 0
                    break;
                end
                
                if o == 1
                    if iter == 1
                        SVsNum = SVsNum + 1;
                        SVsIdx(SVsNum) = j;
                        curId = SVsNum;
                        km = [km; 1];
                    end
                else
                    if iter == 1
                        id = find(SVsIdx(1:SVsNum) == j,1);
                        if ~isempty(id)
                            curId = id;
                        else
                            SVsNum = SVsNum + 1;
                            SVsIdx(SVsNum) = j;
                            curId = SVsNum;
                            km = [km; 1];
                        end
                    end
                end
                
                coff(curId, :) = coff(curId, :) + cur_coeff;
                %re-compute the predicted value for all labels
                pred_v = pred_v + km(curId).* cur_coeff;
            end
        end
    end
    
    tic
    %-------------evaluate model performance on test data-------------------------
    [test_macro_F1_score(run), test_micro_F1_score(run), hammingLoss(run), subsetAccuracy(run), ...
        precision(run), recall(run), F1score(run), rankingLoss(run), oneError(run)] = testEvaluate_kernel_fixed_threshold(SVsIdx(1:SVsNum),coff,SVsNum,scale, gamma);
    clear coff SVsIdx
    testTime(run) = toc;
end
totalTime = toc(tStart);
avgTestTime = mean(testTime);
avgTrainTime = (totalTime - sum(testTime))/times;

clear kernelMatrix
%----------------------output result to file-------------------------------
fid = fopen('FFLT_result.txt','a');
fprintf(fid,'name = yeast, kernel_FFLT, runTimes = %d, epoch = %d, scale = %g, eta = %g, maxIter = %d, gamma = %g\n', times, epoch, scale, eta, maxIterNum, gamma);
fprintf(fid,'precision +std,  recall +std,  F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,\n', mean(precision), std(precision), mean(recall), std(recall), mean(F1score), std(F1score));
fprintf(fid,'macro_F1score +std, micro_F1score +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(test_macro_F1_score), std(test_macro_F1_score), mean(test_micro_F1_score), std(test_micro_F1_score));
fprintf(fid,'hammingloss +std, subsetAccuracy +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f,\n ', mean(hammingLoss), std(hammingLoss), mean(subsetAccuracy), std(subsetAccuracy));
fprintf(fid,'rankingLoss +std, oneErr +std \n');
fprintf(fid,'%.4f, %.4f, %.4f, %.4f \n ', mean(rankingLoss), std(rankingLoss), mean(oneError), std(oneError));
fprintf(fid,'training time [s], testing time [s]\n');
fprintf(fid,'%.4f, %.4f \n\n', avgTrainTime, avgTestTime);
fclose(fid);


%%--------output the result for each run----------------------------------------
fid1 = fopen('appendix/kernel_FFLT_yeast_appendix.txt','a');
for run = 1:times
    fprintf(fid1,' %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n',...
        precision(run), recall(run), F1score(run), test_macro_F1_score(run), test_micro_F1_score(run), ...
        hammingLoss(run), subsetAccuracy(run), rankingLoss(run), oneError(run));
end
fprintf(fid1,'\n');
fclose(fid1);



