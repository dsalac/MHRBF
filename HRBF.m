% Class Definition for Standard/Modified Hermite Radial Basis Function Interpolation
classdef HRBF
    % Standard/Modified Hermite Radial Basis Function Interpolation
    %
    % This class implements both the standard Hermite RBF (HRBF) and Modified Hermite RBF (MHRBF)
    %
    % Usage:
    %   hrbf = HRBF(trainPoints, kernel, kernelParam, polyOrder, matrixType, modPow) % Create the HRBF/MHRBF object
    %   hrbf = hrbf.computeWeights(@myFunction);  % Compute weights for the given function
    %   After computing the weights of MHRBF/HRBF, there are two options: 
    %       1: Evaluate function and derivative(s) at evaluation points
    %       2: Compute interpolation error at given evaluation points
    %   [InterpVals, evalMatrix] = hrbf.evaluateHRBF(evalPoints)  % Evaluate the function and derivatives at evaluation points
    %   err = hrbf.computeError(evalPoints, actualValues, normType) % Computes interpolation error at given evaluation points   
    %
    % Inputs:
    %   trainPoints   - (d X n) array of n interpolation points in d-dimensional space
    %                     - Used to construct the interpolation matrix for weight computation.
    %   param         - Parameter for the radial basis function:
    %                     - For 'GA' (Gaussian) and 'MQ' (Multiquadric), this is the shape parameter.
    %                     - For 'PHS' (Polyharmonic Spline), this is degree of PHS kernel.
    %   kernel        - Type of radial basis function:
    %                     - 'GA'  (Gaussian)
    %                     - 'MQ'  (Multiquadric)
    %                     - 'PHS' (Polyharmonic Spline)
    %   polyOrder     - Order of polynomial augmentation (-1 for none).
    %   matrixType    - Specifies the interpolation matrix type:
    %                     - 'HRBF'  (Standard Hermite RBF)
    %                     - 'MHRBF' (Modified Hermite RBF)
    %   modPow        - Exponent for MHRBF (only relevant for 'MHRBF'. Not required for 'HRBF'):
    %                     - Represents the power term (n) in expressions like (x - x0)^n, (y - y0)^n.
    %
    % Outputs:
    %   hrbf          - HRBF object containing computed weights and methods.
    %
    % Example:
    %   hrbf = HRBF(trainPoints, 'GA', 1, 5, 'MHRBF', 4) 
    %   hrbf = hrbf.computeWeights(@someFunction); % If training using a function handle
    %   hrbf = hrbf.computeWeights(dataVals); % If training using a data values
    %   [interpVals, evalMatrix] = hrbf.evaluateHRBF(evalPoints) % 
    %   err = hrbf.computeError(evalPoints, actualValues, 'Linf')  

    properties (SetAccess=protected)        
        trainPoints    % Interpolation points (d X n)
        trainValues    % Function values at interpolation points
        kernelParam    % Shape parameter in GA/MQ or degree in PHS
        kernel         % RBF kernel type ('MQ', 'GA', 'PHS')
        polyOrder      % Order of polynomial augmentation
        matrixType     % Type of interpolation matrix ('HRBF' or 'MHRBF')
        modPow         % Exponent parameter for MHRBF
    end

    properties (Hidden=true)
        hrbfMatrix  % Interpolation matrix
        nPoly       % Number of polynomial terms
        hrbfWts     % Computed HRBF weights
        rhs         % Right-hand side of linear system
    end

    methods      
        function obj = HRBF(trainPoints, kernel, kernelParam, polyOrder, matrixType, modPow)
        % Create the HRBF/MHRBF object
        %
        % Inputs:
        %   trainPoints - Array of size (d X n) containing n-vertex locations in d-dimensional space
        %   kernelParam - parameter value(s) to use for the Standard/Modified Hermite RBF kernel
        %   kernel      - RBF kernel to use: 'GA', 'MQ', 'PHS'
        %   polyOrder   - Order of the augmenting polynomial 
        %   matrixType  - Type of interpolation matrix ('HRBF' or 'MHRBF')
        %   modPow      - Exponent parameter for MHRBF
        % Output:
        %   obj - The HRBF/MHRBF object

            if size(trainPoints, 1) > size(trainPoints, 2)
                trainPoints = trainPoints';
            end

            obj.trainPoints = trainPoints;
            obj.kernelParam = kernelParam;
            obj.kernel = kernel;
            obj.polyOrder = polyOrder;
            obj.matrixType = matrixType;
            if nargin < 6
                obj.modPow = [];
            else
                obj.modPow = modPow;
            end

            % Augment the matrix if necessary 
            if polyOrder > -1
                obj.nPoly = nchoosek(obj.polyOrder + size(trainPoints,1), obj.polyOrder); % Number of terms in the polynomial  
            end

            % Create derivative matrix dx: first row for zeroth order, next rows for first derivatives
            d = size(trainPoints, 1);
            dx = [zeros(d, 1), eye(d)];

            % Create the base HRBF/MHRBF matrix
            obj.hrbfMatrix = obj.formHRBFMatrix(obj.trainPoints, obj.trainPoints, dx, false);

        end %HRBF/MHRBF

        function obj = computeWeights(obj, f)
        % Determine the weights associated with a given data field. 
        % If any weights already exist they will be overwritten.
        %   
        % Inputs:
        %   f - Either a handle OR a list of function values the same size
        %       as the vertices passed into the MRBF creation function. If a
        %       function handle the call MUST be [f] = c(x, dx). 
        %       Input x is a (dim X 1) array of train points and dx is an (dim X m) array indicating the required derivatives. 
        %       Output is a (1 X m) array containing the derivative values
        %
        % Output:
        %   obj - The updated HRBF/MHRBF object

            [dim, n] = size(obj.trainPoints);
            obj.trainValues = f; % Save the data that was used to compute the weights

            if isa(f, 'function_handle') % If c is a function handle compute all of the values. Otherwise just copy them                                
                switch dim
                    case 1
                        dx = [0 1];
                    case 2
                        dx = [[0 0];[1 0];[0 1]]';
                    case 3
                        dx = [[0 0 0];[1 0 0];[0 1 0];[0 0 1]]';
                    otherwise
                        error('rbf only handles dimensions of 1, 2, or 3.');
                end

                f_vals = zeros(dim+1, n);

                for i=1:n
                    f_vals(:, i) = f(obj.trainPoints(:,i), dx);
                end

                vals = reshape(f_vals', [], 1);
            else
                vals = f;
                if size(vals, 2)>1
                    vals = reshape(vals', [], 1);
                end
                
            end

            % If there is a polynomial augmentation, append the zeros for those terms
            if obj.polyOrder>-1
                vals = [vals; zeros(obj.nPoly, 1)];
            end

            % Store right-hand side of the linear system
            obj.rhs = vals;

            % Compute the weights by solving the linear system
            obj.hrbfWts = obj.hrbfMatrix \ vals;  

        end

        function [interpVals, evalMatrix] = evaluateHRBF(obj, evalPoints, dx)
        % Evaluate a given HRBF/MHRBF. The weights must be computed before this
        % function is called via HRBF.computeWeights
        %
        % Inputs:
        %   evalPoints  - Array of size (dim X n) of locations
        %
        % Output:
        %   interpVals - Array of size (m X n) containing the requested values
        %   evalMatrix - Matrix constructed using the evaluation points


            
            if isempty(obj.hrbfWts)
                error('The weights need to be computed via HRBF.computeWeights before this function is called.\n');
            end

            d = size(evalPoints, 1);
            if nargin==2
                dx = [zeros(d, 1), eye(d)]; % Array of size (dim X m) of function and derivatives to evalaute including zeroth order.
            end


            evalMatrix = obj.formHRBFMatrix(evalPoints, obj.trainPoints, dx, true);
            interpVals = evalMatrix * obj.hrbfWts;

            % Reshape the result so that each location is a column and each
            % derivative is a row
            interpVals = reshape(interpVals, [size(evalPoints, 2), size(dx, 2)])';

        end

        function cnd = conditionNumber(obj)
        % conditionNumber computes the condition number of the HRBF interpolation matrix.
        %
        % This function calculates the condition number of the HRBF interpolation matrix
        % (obj.hrbfMatrix). 
        %
        % Output:
        %   cnd - Condition number of the interpolation matrix.

            cnd = cond(obj.hrbfMatrix);

        end

        function err = computeError(obj, evalPoints, actualValues, normType)
        % Computes interpolation error at given evaluation points
        %
        % Inputs:
        %   evalPoints   - Array of size (dim X n) where function is evaluated
        %   actualValues - Array of size ((dim+1) X n) containing actual function and derivatives values at evalPoints
        %   normType     - String specifying norm type ('L2' or 'Linf')
        %
        % Output:
        %   err - Computed error based on selected norm typ (Array of size ((dim+1) X n))

            % Get predicted values from evaluateHRBF
            interpVals = obj.evaluateHRBF(evalPoints);

            % Compute error based on chosen norm
            diff = abs(actualValues - interpVals);
        
            switch lower(normType)
                case 'l2'
                    err = vecnorm(diff, 2, 2); % L2 norm
                case 'linf'
                    err = vecnorm(diff, inf, 2); % L_Infinity norm
                otherwise
                    error('Invalid normType. Choose "L2" or "Linf".');
            end
        end

    end

    % Helper functions
    methods (Hidden=true)
        function hash = derivativeHash(~, k)
        % DerivativeHash returns a single number giving the requested derivatives
        %
        % Input: 
        %   k - Array of size (d X m) indicating the m-derivatives to return
        %
        % Output:
        %   hash - Array of size (m X 1) with a single number of the requested
        %           derivative. Example values are 100 for x-derivative, 110 for
        %           xy-derivative, 011 for yz, etc.  

            [dim, m] = size(k);
        
            hash = zeros(1, m);
                
            fac = 100;
            for i=1:dim
                hash = hash + k(i, :)*fac;
                fac = fac/10;    
            end  

        end %derivativeHash

        function f = radius(obj, xe, xt, dx)
        % radius returns the radius or derivatives of radius
        %   
        % Input:
        %   xe - Array of size (d X n) OR (d X 1) containing n-locations in d-dimensional space (training points for weight computation or evaluation points for function evaluation)   
        %   xt - Array of size (d X 1) OR (d X n) containing training points 
        %   dx - Array of size (d X 1) indicating the derivative to return
        %
        % Output:
        %   f - The requested values

            n = size(xe, 2);
            m = size(xt, 2);

            if n>1 && m>1
                error('One of the inputs into radius must be a single point.\n');
            end
        
            diff = xe-xt;    
            r = sqrt(sum(diff.^2,1)); % Distances
                
            hash = obj.derivativeHash(dx);
        
            switch hash                
                case 0 % value
                    f = r;
                case 100 %x
                    f = diff(1,:)./r;
                case 200 %xx
                    f = (r.^2 - diff(1,:).^2)./(r.^3);
                case 300 %xxx
                    f = -3*diff(1,:).*(r.^2 - diff(1,:).^2)./(r.^5);
                case 400 %xxxx
                    f = -3*(r.^2 - diff(1,:).^2).*(r.^2 - 5*diff(1,:).^2)./(r.^7);
                case 010 %y
                    f = diff(2,:)./r;
                case 020 %yy
                    f = (r.^2 - diff(2,:).^2)./(r.^3);
                case 030 %yyy
                    f = -3*diff(2,:).*(r.^2 - diff(2,:).^2)./(r.^5);
                case 040 %yyyy
                    f = -3*(r.^2 - diff(2,:).^2).*(r.^2 - 5*diff(2,:).^2)./(r.^7);
                case 110 %xy
                    f = -diff(1,:).*diff(2,:)./(r.^3);
                case 120 %xyy
                    f = diff(1,:).*(-diff(1,:).^2 + 2*diff(2,:).^2)./(r.^5);
                case 210 %xxy
                    f = diff(2,:).*(-diff(2,:).^2 + 2*diff(1,:).^2)./(r.^5);
                case 220 %xxyy
                    f = 2./(r.^3) - 15*(diff(1,:).^2).*(diff(2,:).^2)./(r.^7);
                case 001 %z
                    f = diff(3,:)./r;
                case 002 %zz
                    f = (r.^2 - diff(3,:).^2)./(r.^3);
                case 003 %zzz
                    f = -3*diff(3,:).*(r.^2 - diff(3,:).^2)./(r.^5);
                case 004 %zzz
                    f = -3*(r.^2 - diff(3,:).^2).*(r.^2 - 5*diff(3,:).^2)./(r.^7);
                case 101 %xz
                    f = -diff(1,:).*diff(3,:)./(r.^3);
                case 011 %yz
                    f = -diff(2,:).*diff(3,:)./(r.^3);    
                case 102 %xzz
                    f = diff(1,:).*(-diff(1,:).^2 + 2*diff(3,:).^2)./(r.^5);
                case 201 %xxz
                    f = diff(3,:).*(-diff(3,:).^2 + 2*diff(1,:).^2)./(r.^5);
                case 012 %yzz
                    f = diff(2,:).*(-diff(2,:).^2 + 2*diff(3,:).^2)./(r.^5);
                case 021 %yyz
                    f = diff(3,:).*(-diff(3,:).^2 + 2*diff(2,:).^2)./(r.^5);    
                case 202 %xxzz
                    f = 2./(r.^3) - 15*(diff(1,:).^2).*(diff(3,:).^2)./(r.^7);
                case 022 %yyzz
                    f = 2./(r.^3) - 15*(diff(2,:).^2).*(diff(3,:).^2)./(r.^7);    
                otherwise
                    error('Derivative is unknown.');                       
            end  

        end %radius

    end

    % Matrix formation functions
    methods (Hidden=true)

        function A = baseMatrix(obj, xe, xt, dxIn)  
        % Create a HRBF/MHRBF matrix between two point clouds
        %
        % Input:
        %   xe      - Array of size (d X m) containing n-locations in d-dimensional space where to interpolate (training points for weight computation or evaluation points for function evaluation)  
        %   xt      - Array of size (d X n) containing m-locations in d-dimensional space containing the data (train points) 
        %   dx      - Array containing the requested derivatives. Will be of size (dim X k), where k is the number of derivatives requested.
        %
        % Output:
        %
        %   A - A matrix of size (k*m X n*(dim+1)) that can be multiplied against the
        %       weights in the HRBF/MHRBF approximation.
           
            switch lower(obj.matrixType)
                case 'hrbf'                    
                    rowFunc = @hermiteMatrix_Row; % Standard Hermite Radial Basis Functions
                    md = -1;
                case 'mhrbf'
                    rowFunc = @modHermiteMatrix_Row; % Modified Hermite Radial Basis Functions
                    md = obj.modPow;
                otherwise
                    error('Unknown matrix type.');
            end 

            [dim, m] = size(xe);  % The number of evaluation points (training points for weight computation or new points for function evaluation) 
            n = size(xt, 2);      % The number of train points used to construct the HRBF/MHRBF system 
            k = size(dxIn, 2);      % The number of requested derivatives
        
            nComp = dim + 1;
            A = zeros(k*m, n*nComp);

            off = 1;
            for j=1:k
                [row, dx] = rowFunc(obj, dxIn(:, j));
                for i = 1:m
                    x = (xe(1, i) - xt(1, :));
                    y = (xe(2, i) - xt(2, :));
                    f = obj.kernelEval(xe(:,i), xt, dx);
                    A(off, :) = row(x, y, f, md);
                    off = off + 1;
                end
            end       
        end

        function [row, dx] = hermiteMatrix_Row(obj, dxIn)

            switch obj.derivativeHash(dxIn)
                case 000
	                dx = [0 0 0 ; 0 1 0 ; 1 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(1, :), ...
		                f(3, :), ...
		                f(2, :)));
                
                case 100
	                dx = [0 0 0 ; 1 0 0 ; 1 1 0 ; 2 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 200
	                dx = [0 0 0 ; 2 0 0 ; 2 1 0 ; 3 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 010
	                dx = [0 0 0 ; 0 1 0 ; 0 2 0 ; 1 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 110
	                dx = [0 0 0 ; 1 1 0 ; 1 2 0 ; 2 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 210
	                dx = [0 0 0 ; 2 1 0 ; 2 2 0 ; 3 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 020
	                dx = [0 0 0 ; 0 2 0 ; 0 3 0 ; 1 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 120
	                dx = [0 0 0 ; 1 2 0 ; 1 3 0 ; 2 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                
                case 220
	                dx = [0 0 0 ; 2 2 0 ; 2 3 0 ; 3 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                f(2, :), ...
		                f(4, :), ...
		                f(3, :)));
                otherwise
                    error('Unknown derivative request in modHermiteMatrix_Row');                
            end
        end    

        function [row, dx] = modHermiteMatrix_Row(obj, dxIn)
        % Setup the row equation and required derivatives for the Modified Hermite
            
            switch obj.derivativeHash(dxIn)
                case 000
	                dx = [0 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(1, :), ...
		                x.^(2.*md).*f(1, :), ...
		                y.^(2.*md).*f(1, :)));
                
                case 100
	                dx = [0 0 0 ; 1 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^(md - 1).*y.^md.*(x.*f(2, :) + md.*f(1, :)), ...
		                x.^(2.*md - 1).*(x.*f(2, :) + 2.*md.*f(1, :)), ...
		                y.^(2.*md).*f(2, :)));
                
                case 200
	                dx = [0 0 0 ; 1 0 0 ; 2 0 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(3, :) + 2.*md.*x.^(md - 1).*y.^md.*f(2, :) + md.*x.^(md - 2).*y.^md.*f(1, :).*(md - 1), ...
		                x.^(2.*md).*f(3, :) + 4.*md.*x.^(2.*md - 1).*f(2, :) + 2.*md.*x.^(2.*md - 2).*(2.*md - 1).*f(1, :), ...
		                y.^(2.*md).*f(3, :)));
                
                case 010
	                dx = [0 0 0 ; 0 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^(md - 1).*(y.*f(2, :) + md.*f(1, :)), ...
		                x.^(2.*md).*f(2, :), ...
		                y.^(2.*md - 1).*(y.*f(2, :) + 2.*md.*f(1, :))));
                
                case 110
	                dx = [0 0 0 ; 0 1 0 ; 1 0 0 ; 1 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^(md - 1).*y.^(md - 1).*(md.^2.*f(1, :) + x.*y.*f(4, :) + md.*x.*f(3, :) + md.*y.*f(2, :)), ...
		                x.^(2.*md - 1).*(2.*md.*f(2, :) + x.*f(4, :)), ...
		                y.^(2.*md - 1).*(2.*md.*f(3, :) + y.*f(4, :))));
                
                case 210
	                dx = [0 0 0 ; 0 1 0 ; 1 0 0 ; 1 1 0 ; 2 0 0 ; 2 1 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(6, :) + md.*x.^md.*y.^(md - 1).*f(5, :) + 2.*md.*x.^(md - 1).*y.^md.*f(4, :) + 2.*md.^2.*x.^(md - 1).*y.^(md - 1).*f(3, :) + md.^2.*x.^(md - 2).*y.^(md - 1).*f(1, :).*(md - 1) + md.*x.^(md - 2).*y.^md.*f(2, :).*(md - 1), ...
		                x.^(2.*md).*f(6, :) + 4.*md.*x.^(2.*md - 1).*f(4, :) + 2.*md.*x.^(2.*md - 2).*(2.*md - 1).*f(2, :), ...
		                y.^(2.*md - 1).*(y.*f(6, :) + 2.*md.*f(5, :))));
                
                case 020
	                dx = [0 0 0 ; 0 1 0 ; 0 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(3, :) + 2.*md.*x.^md.*y.^(md - 1).*f(2, :) + md.*x.^md.*y.^(md - 2).*f(1, :).*(md - 1), ...
		                x.^(2.*md).*f(3, :), ...
		                y.^(2.*md).*f(3, :) + 4.*md.*y.^(2.*md - 1).*f(2, :) + 2.*md.*y.^(2.*md - 2).*(2.*md - 1).*f(1, :)));
                
                case 120
	                dx = [0 0 0 ; 0 1 0 ; 0 2 0 ; 1 0 0 ; 1 1 0 ; 1 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(6, :) + 2.*md.*x.^md.*y.^(md - 1).*f(5, :) + md.*x.^(md - 1).*y.^md.*f(3, :) + 2.*md.^2.*x.^(md - 1).*y.^(md - 1).*f(2, :) + md.^2.*x.^(md - 1).*y.^(md - 2).*f(1, :).*(md - 1) + md.*x.^md.*y.^(md - 2).*f(4, :).*(md - 1), ...
		                x.^(2.*md - 1).*(x.*f(6, :) + 2.*md.*f(3, :)), ...
		                y.^(2.*md).*f(6, :) + 4.*md.*y.^(2.*md - 1).*f(5, :) + 2.*md.*y.^(2.*md - 2).*(2.*md - 1).*f(4, :)));
                
                case 220
	                dx = [0 0 0 ; 0 1 0 ; 0 2 0 ; 1 0 0 ; 1 1 0 ; 1 2 0 ; 2 0 0 ; 2 1 0 ; 2 2 0 ; ]';
	                row=@(x, y, f, md)(horzcat(...
		                x.^md.*y.^md.*f(9, :) + 4.*md.^2.*x.^(md - 1).*y.^(md - 1).*f(5, :) + 2.*md.*x.^md.*y.^(md - 1).*f(8, :) + 2.*md.*x.^(md - 1).*y.^md.*f(6, :) + md.^2.*x.^(md - 2).*y.^(md - 2).*f(1, :).*(md - 1).^2 + md.*x.^md.*y.^(md - 2).*(md - 1).*f(7, :) + md.*x.^(md - 2).*y.^md.*(md - 1).*f(3, :) + 2.*md.^2.*x.^(md - 1).*y.^(md - 2).*f(4, :).*(md - 1) + 2.*md.^2.*x.^(md - 2).*y.^(md - 1).*f(2, :).*(md - 1), ...
		                x.^(2.*md).*f(9, :) + 4.*md.*x.^(2.*md - 1).*f(6, :) + 2.*md.*x.^(2.*md - 2).*(2.*md - 1).*f(3, :), ...
		                y.^(2.*md).*f(9, :) + 4.*md.*y.^(2.*md - 1).*f(8, :) + 2.*md.*y.^(2.*md - 2).*(2.*md - 1).*f(7, :)));

                otherwise
                    error('Unknown derivative request in modHermiteMatrix_Row');                
            end
        end
       
        function A = formHRBFMatrix(obj, xe, xt, dx, interpolation)
        % formHRBFMatrix constructs the HRBF/MHRBF interpolation matrix.
        %
        % Inputs:
        %   xe      - (d X m) array containing m evaluation points in d-dimensional space.
        %           These points may be used for weight computation or function evaluation.
        %   xt      - (d X n) array containing n training points in d-dimensional space.
        %   dx      - (dim X k) array specifying the k requested derivatives.
        %   modPow  - Exponent for MHRBF (only relevant for 'MHRBF')
        %
        % Output:
        %   A - (k*m X n*(dim+1)) matrix used in HRBF interpolation.
        %       This matrix is constructed using the kernel function evaluated at
        %       given points and derivatives.

            % The base matrix
            A = obj.baseMatrix(xe, xt, dx);

            % Augmented if necessary
            if obj.polyOrder>-1
                m = size(dx, 2); % Number of derivatives
                n = size(xe, 2); % Number of points
                P = zeros(obj.nPoly, m*n);

                for i=1:m
                    P(:, (i-1)*n+1:i*n) = obj.evalPoly(xe, dx(:,i));
                end

                if interpolation
                    A = [A P'];
                else                    
                    if norm(xe-xt,'inf')>1e-16
                        error('formHRBFMatrix for interpolation==false can only be done between the same two point clouds.\n');
                    end
                    A = [[A P'];[P zeros(obj.nPoly, obj.nPoly)]];
                end
            end
        end

    end % End of Matrix formation functions methods

    % HRBF Kernel methods. The specific kernels have this call:
    %   dx - Array of size (d X m) indicating the m-derivatives to return
    %        OR an array of size (m x 1) indicating the required radial
    %        derivative
    %   r - Distances    
    %
    %   If the function is called with only two inputs then it returns the
    %   limits of the derivatives as the radius goes to zero
    methods (Hidden=true)
        
        function f = kernelEval(obj, xe, xt, dx)
        % Input:
        %   xe - Array of size (d X n) containing location(s) in d-dimensional space 
        %   xt - Array of size (d X 1) containing location of data point
        %   dx - Array of size (d X m) indicating the m-derivatives to return        
        %
        % Output:
        %   f - Array of size (m X n). Each row corresponds to a derivative
        %       in dx and each column corresponds to a point in x
        
            dim = size(xe,1);                % Dimension
            n = max(size(xe,2), size(xt,2)); % Number of points
            m = size(dx, 2);                % Number of derivatives to calculate

            switch lower(obj.kernel)
                case 'ga'
                    kernelFcn = @obj.kernelGA;
                case 'mq'
                    kernelFcn = @obj.kernelMQ;
                case 'phs'
                    kernelFcn = @obj.kernelPHS;
                otherwise
                    error('Unknown RBF type.');
            end                        
            
        
            r = obj.radius(xe, xt, zeros(dim, 1));   % Distance between points
        
            % Assume that all of the base kernel derivatives up to fifth order are needed
            base = zeros(4, n);
            for i=1:6
                base(i, :) = kernelFcn(i-1, r);
            end
        
            hash = obj.derivativeHash(dx); 
  
            f = zeros(m, n);
        
            for i=1:m
                switch hash(i)                                       
                    case 000 %0, 0, 0
	                    f(i, :) = base(1, :);
                    case 100 %1, 0, 0
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                    f(i, :) = base(2, :).*rx;
                    case 200 %2, 0, 0
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                    f(i, :) = base(2, :).*rxx + base(3, :).*rx.^2;
                    case 300 %3, 0, 0
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxx = obj.radius(xe, xt, [3 0 0]');
	                    f(i, :) = base(4, :).*rx.^3 + base(2, :).*rxxx + 3.*base(3, :).*rx.*rxx;
                    case 010 %0, 1, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                    f(i, :) = base(2, :).*ry;
                    case 110 %1, 1, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                    f(i, :) = base(2, :).*rxy + base(3, :).*rx.*ry;
                    case 210 %2, 1, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                    f(i, :) = base(2, :).*rxxy + 2.*base(3, :).*rx.*rxy + base(3, :).*ry.*rxx + base(4, :).*rx.^2.*ry;
                    case 310 %3, 1, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                          rxxx = obj.radius(xe, xt, [3 0 0]');
	                         rxxxy = obj.radius(xe, xt, [3 1 0]');
	                    f(i, :) = base(2, :).*rxxxy + base(5, :).*rx.^3.*ry + 3.*base(3, :).*rx.*rxxy + base(3, :).*ry.*rxxx + 3.*base(3, :).*rxx.*rxy + 3.*base(4, :).*rx.^2.*rxy + 3.*base(4, :).*rx.*ry.*rxx;
                    case 020 %0, 2, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                    f(i, :) = base(2, :).*ryy + base(3, :).*ry.^2;
                    case 120 %1, 2, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                    f(i, :) = base(2, :).*rxyy + base(3, :).*rx.*ryy + 2.*base(3, :).*ry.*rxy + base(4, :).*rx.*ry.^2;
                    case 220 %2, 2, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                         rxxyy = obj.radius(xe, xt, [2 2 0]');
	                    f(i, :) = base(2, :).*rxxyy + 2.*base(3, :).*rxy.^2 + 2.*base(3, :).*rx.*rxyy + 2.*base(3, :).*ry.*rxxy + base(3, :).*rxx.*ryy + base(4, :).*ry.^2.*rxx + base(4, :).*rx.^2.*ryy + base(5, :).*rx.^2.*ry.^2 + 4.*base(4, :).*rx.*ry.*rxy;
                    case 320 %3, 2, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                         rxxyy = obj.radius(xe, xt, [2 2 0]');
	                          rxxx = obj.radius(xe, xt, [3 0 0]');
	                         rxxxy = obj.radius(xe, xt, [3 1 0]');
	                        rxxxyy = obj.radius(xe, xt, [3 2 0]');
	                    f(i, :) = base(2, :).*rxxxyy + 3.*base(3, :).*rx.*rxxyy + 2.*base(3, :).*ry.*rxxxy + 3.*base(3, :).*rxx.*rxyy + 6.*base(3, :).*rxy.*rxxy + base(3, :).*ryy.*rxxx + base(6, :).*rx.^3.*ry.^2 + base(4, :).*ry.^2.*rxxx + 3.*base(4, :).*rx.^2.*rxyy + 6.*base(4, :).*rx.*rxy.^2 + base(5, :).*rx.^3.*ryy + 6.*base(4, :).*rx.*ry.*rxxy + 3.*base(4, :).*rx.*rxx.*ryy + 6.*base(4, :).*ry.*rxx.*rxy + 3.*base(5, :).*rx.*ry.^2.*rxx + 6.*base(5, :).*rx.^2.*ry.*rxy;
                    case 030 %0, 3, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                          ryyy = obj.radius(xe, xt, [0 3 0]');
	                    f(i, :) = base(4, :).*ry.^3 + base(2, :).*ryyy + 3.*base(3, :).*ry.*ryy;
                    case 130 %1, 3, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                          ryyy = obj.radius(xe, xt, [0 3 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                         rxyyy = obj.radius(xe, xt, [1 3 0]');
	                    f(i, :) = base(2, :).*rxyyy + base(5, :).*rx.*ry.^3 + base(3, :).*rx.*ryyy + 3.*base(3, :).*ry.*rxyy + 3.*base(3, :).*rxy.*ryy + 3.*base(4, :).*ry.^2.*rxy + 3.*base(4, :).*rx.*ry.*ryy;
                    case 230 %2, 3, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                          ryyy = obj.radius(xe, xt, [0 3 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                         rxyyy = obj.radius(xe, xt, [1 3 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                         rxxyy = obj.radius(xe, xt, [2 2 0]');
	                        rxxyyy = obj.radius(xe, xt, [2 3 0]');
	                    f(i, :) = base(2, :).*rxxyyy + 2.*base(3, :).*rx.*rxyyy + 3.*base(3, :).*ry.*rxxyy + base(3, :).*rxx.*ryyy + 6.*base(3, :).*rxy.*rxyy + 3.*base(3, :).*ryy.*rxxy + base(6, :).*rx.^2.*ry.^3 + 3.*base(4, :).*ry.^2.*rxxy + base(4, :).*rx.^2.*ryyy + 6.*base(4, :).*ry.*rxy.^2 + base(5, :).*ry.^3.*rxx + 6.*base(4, :).*rx.*ry.*rxyy + 6.*base(4, :).*rx.*rxy.*ryy + 3.*base(4, :).*ry.*rxx.*ryy + 6.*base(5, :).*rx.*ry.^2.*rxy + 3.*base(5, :).*rx.^2.*ry.*ryy;
                    case 330 %3, 3, 0
	                            ry = obj.radius(xe, xt, [0 1 0]');
	                           ryy = obj.radius(xe, xt, [0 2 0]');
	                          ryyy = obj.radius(xe, xt, [0 3 0]');
	                            rx = obj.radius(xe, xt, [1 0 0]');
	                           rxy = obj.radius(xe, xt, [1 1 0]');
	                          rxyy = obj.radius(xe, xt, [1 2 0]');
	                         rxyyy = obj.radius(xe, xt, [1 3 0]');
	                           rxx = obj.radius(xe, xt, [2 0 0]');
	                          rxxy = obj.radius(xe, xt, [2 1 0]');
	                         rxxyy = obj.radius(xe, xt, [2 2 0]');
	                        rxxyyy = obj.radius(xe, xt, [2 3 0]');
	                          rxxx = obj.radius(xe, xt, [3 0 0]');
	                         rxxxy = obj.radius(xe, xt, [3 1 0]');
	                        rxxxyy = obj.radius(xe, xt, [3 2 0]');
	                       rxxxyyy = obj.radius(xe, xt, [3 3 0]');
	                    f(i, :) = 6.*base(4, :).*rxy.^3 + base(2, :).*rxxxyyy + base(7, :).*rx.^3.*ry.^3 + base(5, :).*ry.^3.*rxxx + base(5, :).*rx.^3.*ryyy + 3.*base(3, :).*rx.*rxxyyy + 3.*base(3, :).*ry.*rxxxyy + 3.*base(3, :).*rxx.*rxyyy + 9.*base(3, :).*rxy.*rxxyy + 3.*base(3, :).*ryy.*rxxxy + base(3, :).*rxxx.*ryyy + 9.*base(3, :).*rxxy.*rxyy + 3.*base(4, :).*ry.^2.*rxxxy + 3.*base(4, :).*rx.^2.*rxyyy + 9.*base(4, :).*rx.*ry.*rxxyy + 3.*base(4, :).*rx.*rxx.*ryyy + 18.*base(4, :).*rx.*rxy.*rxyy + 9.*base(4, :).*rx.*ryy.*rxxy + 9.*base(4, :).*ry.*rxx.*rxyy + 18.*base(4, :).*ry.*rxy.*rxxy + 3.*base(4, :).*ry.*ryy.*rxxx + 9.*base(4, :).*rxx.*rxy.*ryy + 9.*base(5, :).*ry.^2.*rxx.*rxy + 9.*base(5, :).*rx.*ry.^2.*rxxy + 9.*base(5, :).*rx.^2.*ry.*rxyy + 9.*base(5, :).*rx.^2.*rxy.*ryy + 9.*base(6, :).*rx.^2.*ry.^2.*rxy + 18.*base(5, :).*rx.*ry.*rxy.^2 + 3.*base(6, :).*rx.*ry.^3.*rxx + 3.*base(6, :).*rx.^3.*ry.*ryy + 9.*base(5, :).*rx.*ry.*rxx.*ryy;                 
                    otherwise
                        error('Derivative (%d,%d) is unknown.', dx(1, m), dx(2, m));
                end
        
                % For any zero-distance points adjust to the limit of the
                % derivative
                if hash(i) > 0
                    lm = kernelFcn(dx(:, i), []);
                    f(i, r < 1e-12) = lm;
                end
        
            end            

        end

        function phi = kernelGA(obj, dx, r)
        % kernelGA computes the Gaussian RBF kernel and its derivatives.
        %
        % Inputs:
        %   dx - Specifies the order of the derivative to be computed.
        %        If dx is a matrix, it indicates partial derivatives in multiple dimensions.
        %   r  - Distance values where the kernel should be evaluated.
        %
        % Output:
        %   phi - Computed kernel values or derivative evaluations.
        %
        % The Gaussian kernel is defined as:
        %       phi(r) = exp(-(r/a)^2)
        % where 'a' is a scaling parameter (kernel width).
        %
        % Derivatives:
        %   phi'(r)  = -2r * exp(-(r/a)^2) / a^2
        %   phi''(r) = -2(a^2 - 2r^2) * exp(-(r/a)^2) / a^4

            a = obj.kernelParam;

            if nargin==1 || isempty(r)
                [~, m] = size(dx);            
                phi = zeros(m,1);                       
                hash = obj.derivativeHash(dx);
            
                I = find(hash==200 | hash==020 | hash==002);
                if ~isempty(I)
                    phi(I) = -2/(a*a);
                end
            else
                phi0 = exp(-(r./a).^2);
            
                switch dx
                    case 0
                        phi = phi0;
                    case 1
                        phi = -2*r.*phi0./(a*a);
                    case 2
                        phi = -2*(a*a - 2*r.*r).*phi0./(a^4);
                    case 3
                        phi = (12*a*a*r - 8*r.*r.*r).*phi0./(a^6);
                    case 4
                        phi = (4*(3*(a^4) - 12*a*a*r.*r + 4*(r.^4))).*phi0./(a^8);
                    case 5
                        phi = (-8*(15*power(a,4)*r - 20*power(a,2)*power(r,3) + 4*power(r,5))).*phi0./power(a,10);
                    case 6
                        phi = (-8*(15*power(a,6) - 90*power(a,4)*power(r,2) + 60*power(a,2)*power(r,4) - 8*power(r,6))).*phi0./power(a,12);
                    otherwise
                        error('It is not possible to evaluate the derivative of order %d\n', k);
                end
            end
        end %kernelGA

        function phi = kernelMQ(obj, dx, r)
        % kernelMQ computes the Multiquadric (MQ) RBF kernel and its derivatives.
        %
        % Inputs:
        %   dx - Specifies the order of the derivative to be computed.
        %        If dx is a matrix, it indicates partial derivatives in multiple dimensions.
        %   r  - Distance values where the kernel should be evaluated.
        %
        % Output:
        %   phi - Computed kernel values or derivative evaluations.
        %
        % The Multiquadric kernel is defined as:
        %       phi(r) = sqrt(r^2 + a^2)
        % where 'a' is a scaling parameter (kernel width).
        %
        % Derivatives:
        %   phi'(r)  = r / sqrt(r^2 + a^2)
        %   phi''(r) = (a^2) / (r^2 + a^2)^(3/2)

            a = obj.kernelParam;

            if nargin==1 || isempty(r)
                [~, m] = size(dx);           
                phi = zeros(m,1);                       
                hash = obj.derivativeHash(dx);
            
                I = find(hash==200 | hash==020 | hash==002);
                if ~isempty(I)
                    phi(I) = 1/a;
                end

                I = find(hash==400 | hash==040 | hash==004);
                if ~isempty(I)
                    phi(I) = -3/(a^3);
                end

                I = find(hash==220 | hash==202 | hash==022);
                if ~isempty(I)
                    phi(I) = -1/(a^3);
                end
            else
                switch dx
                    case 0
                        phi = sqrt(r.^2 + a^2);
                    case 1
                        phi = r./sqrt(r.^2 + a^2);
                    case 2
                        phi = (a^2)./power(r.^2 + a^2, 1.5);
                    otherwise
                        error('It is not possible to evaluate the derivative of order %d.\n', k);
                end            
            end
        end %kernelMQ
    
        function phi = kernelPHS(obj, dx, r)
        % kernelPHS computes the Polyharmonic Spline (PHS) kernel and its derivatives.
        %
        % Inputs:
        %   dx - Specifies the order of the derivative to be computed.
        %        If dx is a matrix, it indicates partial derivatives in multiple dimensions.
        %   r  - Distance values where the kernel should be evaluated.
        %
        % Output:
        %   phi - Computed kernel values or derivative evaluations.
        %
        % The Polyharmonic Spline kernel is defined as:
        %       phi(r) = r^m
        % where 'm' is the degree of the kernel.
        %
        % Derivatives:
        %   phi'(r)  = m * r^(m-1)
        %   phi''(r) = m * (m-1) * r^(m-2)

            m = obj.kernelParam;

            if nargin==1 || isempty(r)
                [~, g] = size(dx);
                phi = zeros(g,1);                
                hash = obj.derivativeHash(dx);
        
                if m < 3
                    I = find(hash==300 | hash==030 | hash==003);
                    if ~isempty(I)
                        phi(I) = Inf;
                    end
                end
                
                if m < 2
                    I = find(hash==200 | hash==020 | hash==002);   
                    if ~isempty(I)
                        phi(I) = Inf;
                    end
                end

            else                
                switch dx
                    case 0
                        phi = r.^m;
                    case 1            
                        phi = m*(r.^(m-1));
                    case 2
                        if m<2
                            phi = 0;
                        else
                            phi = (m-1)*m*(r.^(m-2));
                        end
                    otherwise
                        error('It is not possible to evaluate the derivative of order %d.\n', k);
                end
            end
        end %kernelPHS
    end

    % Augmenting polynomial functions
    methods (Hidden=true)

        function f = evalPoly(obj, x, dx)
        % evalPoly evaluates polynomial functions for given input points and their derivatives.
        %
        % Inputs:
        %   x  - Array of points for which the polynomial basis functions need to be evaluated.
        %        This is a (d X n) matrix where d is the dimensionality of the space (1, 2, or 3).
        %   dx - Derivative orders for each dimension (1 X d array).
        %
        % Output:
        %   f  - Polynomial basis functions evaluated at the input points.
        
            dim = size(x, 1);

            switch dim
                case 1
                    f = obj.monomial1D(x, dx);
                case 2
                    f = obj.monomial2D(x, dx);
                case 3
                    f = obj.monomial3D(x, dx);
                otherwise
                    error('Dimension can only be from 1 to 3');
            end
        end

    end

    % Polynomials based on monomials
    methods (Hidden=true)
        
        % Evaluate 1D monomial basis functions
        function f = monomial1D(obj, x, dx) 
        % monomial1D evaluates 1D polynomial basis functions for given points and derivatives.
        %
        % Inputs:
        %   x  - Array of input points (1 X n).
        %   dx - Derivative order for the polynomial evaluation (1 X 1).
        %
        % Output:
        %   f  - Polynomial basis functions evaluated at the input points for 1D.
        %        This is an array with rows representing different monomial terms.
        
            n = length(x);
        
            f = zeros(obj.nPoly+1, n);

            for i=dx:obj.polyOrder
                f(i+1, :) = prod(i-dx+1:i)*x.^(i-dx(1));
            end
        
        end %Poly1D
        
        % Evaluate 2D monomial basis functions
        function f = monomial2D(obj, x, dx)
        % monomial2D evaluates 2D polynomial basis functions for given points and derivatives.
        %
        % Inputs:
        %   x  - Array of 2D input points (2 X n), where each row corresponds to a different dimension.
        %   dx - Derivative order for each dimension (1 X 2 array).
        %
        % Output:
        %   f  - Polynomial basis functions evaluated at the input points for 2D.
        %        This is an array with rows representing different 2D monomial terms.
        
        
            fx = obj.monomial1D(x(1, :), dx(1));
            fy = obj.monomial1D(x(2, :), dx(2));
        
            n = size(x, 2);
        
            f = zeros(obj.nPoly, n);
            iter = 1;
            for i=0:obj.polyOrder
                for j=0:obj.polyOrder - i            
                    f(iter, :) = fx(i+1, :).*fy(j+1, :);
                    iter = iter + 1;
                end
            end
        
        end %Poly2D
        
        % Evaluate 3D monomial basis functions
        function f = monomial3D(obj, x, dx)
        % monomial3D evaluates 3D polynomial basis functions for given points and derivatives.
        %
        % Inputs:
        %   x  - Array of 3D input points (3 X n), where each row corresponds to a different dimension.
        %   dx - Derivative order for each dimension (1 X 3 array).
        %
        % Output:
        %   f  - Polynomial basis functions evaluated at the input points for 3D.
        %        This is an array with rows representing different 3D monomial terms.
        
        
            fx = obj.monomial1D(x(1, :), dx(1));
            fy = obj.monomial1D(x(2, :), dx(2));
            fz = obj.monomial1D(x(3, :), dx(3));
        
            n = size(x, 2);
        
            f = zeros(obj.nPoly, n);
            iter = 1;
            for i=0:obj.polyOrder
                for j=0:obj.polyOrder-i
                    for k=0:obj.polyOrder-i-j
                        f(iter, :) = fx(i+1, :).*fy(j+1, :).*fz(k+1, :);
                        iter = iter + 1;
                    end
        
                end
            end
        
        end %Poly3D
    end
end % HRBF Class