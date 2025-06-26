function [f] = Eq13(x, dx)
%Eq13 - Return function and derivatives
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
                f(i, :) = cos(4.*y) + sin(6.*x) + sin(3.*x + 2.*y);
            case 100 % dx
                f(i, :) = 6.*cos(6.*x) + 3.*cos(3.*x + 2.*y);
            case 010 % dy
                f(i, :) = 2.*cos(3.*x + 2.*y) - 4.*sin(4.*y);
            case 001 % dz
                f(i, :) = 0;
            case 200
                f(i, :) = - 36.*sin(6.*x) - 9.*sin(3.*x + 2.*y);
            case 020
                f(i, :) = - 16.*cos(4.*y) - 4.*sin(3.*x + 2.*y);
            case 002
                f(i, :) = 0;
            case 110
                f(i, :) = -6.*sin(3.*x + 2.*y);
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
end %Eq13
