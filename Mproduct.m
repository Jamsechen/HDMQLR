clear all;
n = 500; p = 500;q = 100; X = randn(n,p);B = randn(p,q);tic;X'*(X*B);toc;
Y = randn(n,q);
tic;triu(X'*X)*B;toc;
tic;X*B;toc;
tic;X'*(X*B);toc;

% tic;
% S = zeros(p,p);
% A = zeros(p,q);
% for i=1:p
%     for j=1:p
%         if (j>i)
%           A(i,:) =  A(i,:) + X(:,i)'*(X(:,j)*B(j,:));
%         end
%     end
% end
% toc;