
%% A:=(X'X+sigma*I) = LU=LL'
% if AB=C, then B=A\C
function [L,U] = factor(X, sigma)
    [n, p] = size(X);
    if ( n >= p )    % if skinny
       L = chol( X'*X + sigma*speye(p), 'lower' );
    else            % if fat
       L = chol( speye(n) + 1/sigma*(X*X'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end