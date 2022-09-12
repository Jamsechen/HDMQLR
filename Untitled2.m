clear all; 
n = 500; p = 800; q = 2;
a = randn(n,p); 
rho = 1;
A =a'*a + rho*eye(p); 
B = randn(p,q);
 tic;
 [L,U]  = factor(a, rho);
if (n >= p)    % if  X is skinny
    X_new = U\(L\B);
else            % if X is fat
    X_new = B/rho - (a'*(U\(L\(a*B))))/rho^2;
end
toc;
norm(A*X_new-B,'fro')
tic;
for i=1:q
    X_new1(:,i) = CG(A,B(:,i));
end
toc;
norm(A*X_new1-B,'fro')