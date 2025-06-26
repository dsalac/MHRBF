function [f] = reinit_cassini(x, dx)
%reinit_cassini - Return function and derivatives of a circle with radius 0.5
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
                f(i, :) = ((x - 1./2).^2 + y.^2).*((x + 1./2).^2 + y.^2) - 1171615957919061./18014398509481984;
            case 100 % dx
                f(i, :) = x.*(4.*x.^2 + 4.*y.^2 - 1);
            case 010 % dy
                f(i, :) = y.*(4.*x.^2 + 4.*y.^2 + 1);
            case 001 % dz
                f(i, :) = 0;
            case 200
                f(i, :) = 12.*x.^2 + 4.*y.^2 - 1;
            case 020
                f(i, :) = 4.*x.^2 + 12.*y.^2 + 1;
            case 002
                f(i, :) = 0;
            case 110
                f(i, :) = 8.*x.*y;
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
end %reinit_cassini
