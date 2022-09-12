
%% Solution for ||B - G||_F^2/2 + lambda*||B||_{1,2}
%Output: B--Solution; supp--the index of nonzero-row in B


%--------------------Main Code-------------------------------
function [B,supp] = Shrinkage_Block(G,lambda)
[p,q] = size(G);
B     = zeros(p,q);
supp  = [ ];
for i=1:p
    G2 = norm(G(i,:),2);
    if G2>lambda
       B(i,:) = (1-lambda/G2)*G(i,:);
       supp = [supp i];
    end
end