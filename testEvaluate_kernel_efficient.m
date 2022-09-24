
function [macro, micro, hammingloss, subsetAccuracy, precision, recall, F1score, rankingLoss,oneErr] = testEvaluate_kernel_efficient(SVsIdx,coff,SVsNum,scale)
global trainData testData testLabel


%-------------evaluate model performance on test data-------------------------
N = size(testData,1);
L = size(testLabel,2);

% chunk the testing data into smaller size and proceed the testing sequentially to prevent Matlab Out-of-Memory
chunk = 1000; 
km = zeros(SVsNum,N);
for i = 1:ceil(N/chunk)
    ind_start = 1+(i-1)*chunk;
    if i*chunk<N
        ind_end = i*chunk;
    else
        ind_end = N;
    end
    [km(:,ind_start:ind_end),~]= rbfkernel(trainData(SVsIdx,:)',testData(ind_start:ind_end,:)',scale);
end

F = 0;
tp_sum = 0;
fp_sum = 0;
fn_sum = 0;
tn_sum = 0;
hammingloss = 0;
same = 0;

pred_v = zeros(N,L);
pred_lv = zeros(N,L);
threshold = km'* coff(1:SVsNum,L+1);

for i = 1:L
    
    orig = testLabel(:, i);
    pred_v(:,i) = km'* coff(1:SVsNum,i);
    pred = pred_v(:,i) > threshold;
    pred_lv(:,i) = pred; 
    
    tp = full(sum(orig == +1 & pred == +1));
    fn = full(sum(orig == +1 & pred == 0));
    tn = full(sum(orig == 0 & pred == 0));
    fp = full(sum(orig == 0 & pred == +1));
    
    this_F = 0;
    if tp ~= 0 || fp ~= 0 || fn ~= 0
        this_F = (2*tp) / (2*tp + fp + fn);
    end
    F = F + this_F;
    
    tp_sum = tp_sum + tp;
    fp_sum = fp_sum + fp;
    fn_sum = fn_sum + fn;
    tn_sum = tn_sum + tn;
    
    pred = 2 * pred - 1;
    orig = 2 * orig - 1;
    
    hammingloss = hammingloss + N - pred'* orig;
    same = same + pred.* orig;
end

macro = F/L;
micro = (2*tp_sum) / (2*tp_sum + fp_sum + fn_sum);
hammingloss = hammingloss / (2*N*L);
subsetAccuracy = sum(same == L)/ N;

[rankingLoss, oneErr]= calRankErrLoss(testLabel,pred_v);
[precision, recall, F1score] = calPrecisionRecall (testLabel, pred_lv, N);

end


