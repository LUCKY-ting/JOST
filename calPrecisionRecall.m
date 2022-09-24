function [precision, recall, F1score] = calPrecisionRecall(groundTrue, pred_lv, N)

precision = 0;
recall = 0;
for i = 1:N
    if sum(pred_lv(i,:)) ~= 0
        tp_precision =  groundTrue(i,:) * pred_lv(i,:)'/ sum(pred_lv(i,:));
    else
        if sum(groundTrue(i,:)) == 0
             tp_precision = 1;
        else
             tp_precision = 0;
        end
    end
    if sum(groundTrue(i,:))~= 0
        tp_recall = groundTrue(i,:) * pred_lv(i,:)'/ sum(groundTrue(i,:));
    else
        if sum(pred_lv(i,:)) == 0
            tp_recall = 1;
        else
            tp_recall = 0;
        end
    end
    precision = precision + tp_precision;
    recall = recall + tp_recall;
end

precision = precision / N;
recall = recall / N;
F1score = 2 * precision * recall / (precision + recall);

end
