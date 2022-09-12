function [Z, Sup_Z] = shrinkage(B,mu)
   [p,q] = size(B);
   Z     = zeros(p,q);
   Sup_Z = [ ];
for i=1:p
    a = B(i,:);
    b = norm(a,2);
    if 1-mu(i)/b>0
       Z(i,:) = (1-mu(i)/b)*a;
       Sup_Z  = [Sup_Z  i];
   end
end