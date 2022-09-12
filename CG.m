function x = CG(A,b)
tol = 1e-12;
r   = b + A*b;
w   = -r;
z   = A*w;
s   = w'*z;
t   = (r'*w)/s;
x   = -b + t*w;
for k = 1:numel(b);
    r = r - t*z;
    if( norm(r) < tol )
        return;
    end
    B = (r'*z)/s;
    w = -r + B*w;
    z = A*w;
    s = w'*z;
    t = (r'*w)/s;
    x = x + t*w;
end