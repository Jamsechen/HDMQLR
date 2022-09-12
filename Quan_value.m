function value = Quan_value(E,tau)
[n,q] = size(E);
value = 0;
for i=1:n
    for j=1:q
     if E(i,j)>=0
         value = value+tau*E(i,j);
     else
         value = value+(tau-1)*E(i,j);
     end
    end
end
