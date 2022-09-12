function x=Cong(a,b,c,x0)
n = length(a);
if nargin<4
    x0=zeros(n,1);
end
if nargin<3
    c=10^(-3);
end
r0=b-a*x0;
r=r0;
d=r0;
m=zeros(n,1);
p=zeros(n,1);
x=x0;
for k=0:n-1
    m(k+1)=r'*r/(d'*a*d);
    x=x+m(k+1)*d;
    r=b-a*x;
    if (norm(r,inf)<=c)|(k+1==n)
        break;
    end
    p(k+1)=norm(r)^2/norm(r0)^2;
    d=r+p(k+1)*d;
    r0=r;
end
