function [lSQvalue,Grad] = lSmooth_Quan_value(X,Y,B,tau,kappa)

[n,q]    = size(Y);
A        = (Y-X*B)/n;
lSQvalue = 0;
Grad     = zeros(n,q);
for i=1:n
    for j=1:q
        u = A(i,j);
     if (u<(tau-1)*kappa)
        lSQvalue  = lSQvalue + (tau-1)*u - kappa*(1-tau)^2/2;
        Grad(i,j) = tau - 1;
     elseif (u<=tau*kappa)
        lSQvalue  = lSQvalue + u^2/(2*kappa);
        Grad(i,j) = u/kappa;
     else
        lSQvalue  = lSQvalue + tau*u - kappa*tau^2/2;
        Grad(i,j) = tau;
     end
    end
end
