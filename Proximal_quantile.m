function C = Proximal_quantile(B,mu,tau)
[p,q] = size(B);
C     = zeros(p,q);
for i=1:p
for j=1:q
    a  = B(i,j);
    if a>mu*tau
       C(i,j) = a - mu*tau;
    end
    if a<mu*(tau-1)
       C(i,j) = a - mu*(tau-1);
    end
end
end
