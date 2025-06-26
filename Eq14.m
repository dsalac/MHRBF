function [f] = Eq14(x, dx)
%Eq14 - Return function and derivatives
%
% Input:
%   x - Evaluation point. Size is (dim X 1)
%   dx - Array of size (dim X m) containing the requested derivatives
%
% Output:
%   f - Array of size (m X n) with each row containing the requested
%   derivative

    [dim, ~] = size(x);
    m = size(dx, 2);
    
    hash = zeros(1, m);
    
    fac = 100;
    for i = 1:dim
        hash = hash + dx(i, :) .* fac;
        fac = fac / 10;
    end
    
    f = zeros(m,size(x, 2));
    if dim>1
        y = x(2, :);
    else
        y = 0;
    end
    if dim>2
        z = x(3, :);
    else
        z = 0;
    end
    x = x(1, :);
    for i = 1:m
        switch hash(i)
            case 000
                f(i, :) = x.^2.*(x.^4./3 - (21.*x.^2)./10 + 4) + x.*y + y.^2.*(4.*y.^2 - 4);
            case 100 % dx
                f(i, :) = 8.*x + y - (42.*x.^3)./5 + 2.*x.^5;
            case 010 % dy
                f(i, :) = x - 8.*y + 16.*y.^3;
            case 001 % dz
                f(i, :) = 0;
            case 200
                f(i, :) = 10.*x.^4 - (126.*x.^2)./5 + 8;
            case 020
                f(i, :) = 48.*y.^2 - 8;
            case 002
                f(i, :) = 0;
            case 110
                f(i, :) = 1;
            case 101
                f(i, :) = 0;
            case 011
                f(i, :) = 0;
            case 111
                f(i, :) = 0;
            otherwise
                error('Derivative calculation has not yet been set up');
        end
    end
end %Eq14
