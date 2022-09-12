function [uSQvalue,Grad] = uSmooth_Quan_value(X,Y,B,tau,kappa)

[n,q]    = size(Y);
A        = (Y-X*B)/n;
uSQvalue = 0;
Grad     = zeros(n,q);
for i=1:n
    for j=1:q
        u = A(i,j);
     if (u<-kappa)
        uSQvalue  = uSQvalue + (tau-1)*u;
        Grad(i,j) = tau - 1;
     elseif (u<=kappa)
        uSQvalue  = uSQvalue + u^2/(4*kappa) + (tau-1/2)*u + kappa/4;
        Grad(i,j) = u/(2*kappa) + tau - 1/2;
     else
        uSQvalue  = uSQvalue + tau*u;
        Grad(i,j) = tau;
     end
    end
end
