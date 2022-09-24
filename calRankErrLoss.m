
function [rloss,oneError] = calRankErrLoss(groundTrue,pred_v)

rloss = 0;
N = size(groundTrue, 1);
oneError = 0;
[~,Idx] = max(pred_v,[],2);

for i = 1:N
    rele = find(groundTrue(i,:));
    irrele = find(~(groundTrue(i,:)));
    if size(rele,2) > 0 && size(irrele, 2) > 0
        misorder = 0;
        for j = 1: size(rele,2)
            for k = 1: size(irrele, 2)
                if pred_v(i,rele(j)) <= pred_v(i,irrele(k))
                    misorder = misorder + 1;
                end
            end
        end
        rloss = rloss + misorder / (size(rele,2)*size(irrele,2));
        
        if groundTrue(i,Idx(i)) == 0
            oneError = oneError + 1;
        end
    end
end
rloss = rloss / N;
oneError = oneError / N;

end

