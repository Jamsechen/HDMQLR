function Z = keep_largest(Z, K)
[p,q] = size(Z,1);
for i=1:p
normZ(i) = norm(Z(i,:),'fro');
end
    [val Ind] = sort(normZ, 'descend');
    Z(Ind(K+1:end),:) = zeros(1,q);
end