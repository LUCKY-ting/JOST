clc
clear
global testData testLabel L;
load('datasets/bibtex-train.mat');
load('datasets/bibtex-test.mat');
[n,d] = size(trainData);
L = size(trainLabel,2);

trainData = sparse(trainData);
trainLabel = sparse(trainLabel);
testData = sparse(testData);
testLabel = sparse(testLabel);

epoch = 1;
times = 20;  % run 20 times for calculating mean accuracy

%%hyperparameters for SFLT
delta = 2.^(1);
eta = 2.^(-3);
maxIter = 40;
gamma = 0.5; % fixed threshold

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

tStart = tic;
for run = 1:times
    index = randperm(n);
    w = SFLT_train_sparse(trainData',trainLabel',index, epoch, eta, delta, maxIter, gamma);
    tic
    [test_macro_F1_score(run), test_micro_F1_score(run), hammingLoss(run), subsetAccuracy(run),...
        precision(run), recall(run), F1score(run), rankingLoss(run), oneError(run)] = testEvaluate_fixed_threshold(w, gamma);
    testTime(run) = toc;
end
totalTime = toc(tStart);
avgTestTime = mean(testTime);
avgTrainTime = (totalTime - sum(testTime))/times;

%%--------------------output result to file------------------------
fid = fopen('SFLT_result.txt','a');
fprintf(fid,'name = bibtex, SFLT \n');
fprintf(fid,'runTimes = %d, epoch = %d, delta = %g, eta = %g, maxIter = %d, gamma = %g \n', times, epoch, delta, eta, maxIter, gamma);
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

fid1 = fopen('appendix/SFLT_bibtex_appendix.txt','a');
for run = 1:times
    fprintf(fid1,' %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f \n',...
        precision(run), recall(run), F1score(run), test_macro_F1_score(run), test_micro_F1_score(run), ...
        hammingLoss(run), subsetAccuracy(run), rankingLoss(run), oneError(run));
end
fprintf(fid1,'\n');
fclose(fid1);


