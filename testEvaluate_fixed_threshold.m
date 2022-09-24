
function [macro, micro, hammingloss, subsetAccuracy, precision, recall, F1score, rankingLoss, oneErr] = testEvaluate_fixed_threshold(w, gamma)
global testData testLabel L;
%-------------evaluate model performance on test data-------------------------
N = size(testData,1);
F = 0;
tp_sum = 0;
fp_sum = 0;
fn_sum = 0;
tn_sum = 0;
hammingloss = 0;
same = 0;

pred_v = zeros(N,L);
pred_lv = zeros(N,L);
threshold = gamma;

for i = 1:L
  
    orig = testLabel(:, i);
    pred_v(:,i) = testData * w(:,i);
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



