function [phi, cp, H] = reinit(reinitPts, dataPts, rbfData)
%REINIT Reinitialize a function to a SDF by computing the closest point
%
% Inputs:
%   reinitPts - Points to reinitialize
%   dataPts - Points where the data is defined
%   rbfData - Struct with basic information
%       epsilon: Kernel parameter
%       kernelPower: Power of the modified MHRBF kernel
%       kernel: 'ga' for Gaussian
%       polyOrder: Augmenting polynomial order
%       numHRBFPoints: Number of points to use for the RBF
%       matType: 'MHRBF' or 'HRBF'
%       phi0: Function handle to level set that will be reinitialized. See
%               reinit_cassini for an example.
%       h: Average spacing of the mesh
%
% Output:
%   phi - SDF of f
%   cp - Closest points
%   H - Curvature at closest points

    % Form the triangularization and cell-centers
    DT = delaunayTriangulation(dataPts);
    TR = triangulation(DT.ConnectivityList, DT.Points);
    IC = incenter(DT);
    nCells = size(DT.ConnectivityList, 1);
        
    % Set the additional data
    rbfData.dataPts = dataPts;
    rbfData.kernelParam = 1/rbfData.epsilon;    
    rbfData.hrbf = cell(nCells, 1);

    nVerts = size(reinitPts, 1);
    phi = inf(nVerts, 1);
    cp = inf(nVerts, size(reinitPts, 2));
    H = inf(nVerts, 1);
    h = rbfData.h;
    phi0 = rbfData.phi0;

    % Now reinitialize
    for i = 1:nVerts  
        y = reinitPts(i, :);

        % Take a small step towards the true closest point.
        %   In the situation there is more than one valid closest point
        %   make sure that we're going to the correct one.
        y = y + 0.01*h*(rbfData.cp(i, :) - y); 

        % Find the closest point and store in y
        [y, rbfData] = chopp(reinitPts(i, :), y, TR, IC, rbfData, 1e-3*h*h);                

        p0 = phi0(reinitPts(i,:)', [0;0]);
        phi(i) = sign(p0)*sqrt(sum((reinitPts(i,:) - y).^2));
        cp(i, :) = y;

        cellID = pointLocation(TR, y);  
        dx = [[0 0];[1 0];[0 1];[2 0];[0 2];[1 1]]';
        rbfData = setRBF(cellID, IC, rbfData);
        f = rbfData.hrbf{cellID}.evaluateHRBF(y', dx);
        fx = f(2); fy = f(3);
        fxx = f(4); fyy = f(5); fxy = f(6);
        H(i) = (fxx*fy*fy + fyy*fx*fx - 2*fxy*fx*fy)/((fx*fx + fy*fy)^1.5);
    end
end

function rbfData = setRBF(cellID, IC, rbfData)
    if isempty(rbfData.hrbf{cellID})
        warning('off');
        [~, I] = pdist2(rbfData.dataPts, IC(cellID, :), 'euclidean', 'Smallest', rbfData.numHRBFPoints);
        rbfData.hrbf{cellID} = HRBF(rbfData.dataPts(I, :), rbfData.kernel, ...
            rbfData.kernelParam, rbfData.polyOrder, rbfData.matType, rbfData.kernelPower);
        rbfData.hrbf{cellID} = rbfData.hrbf{cellID}.computeWeights(rbfData.phi0);
        warning('on');
    end
end

function [y, rbfData] = chopp(x0, y, TR, IC, rbfData, tol)
    % Modified version of "SOME IMPROVEMENTS OF THE FAST MARCHING METHOD"
    % by Chopp
    
    iter = 0;
    cellID = pointLocation(TR, y);
    d1 = inf;
    d2 = inf;
    h = rbfData.h;
    
    while sqrt(dot(d1, d1) + dot(d2, d2)) > tol
    
        iter = iter + 1;
        if iter == 1000            
            error('Too many iterations!');
        end
        rbfData = setRBF(cellID, IC, rbfData);
        f = rbfData.hrbf{cellID}.evaluateHRBF(y');            
        p = f(1);
        g = f(2:end)';            

        % Don't let either step be too large. This will slow the 
        %   calculation down but it's more stable.
        d1 = -p*g./dot(g,g);
        if norm(d1) > 2*h
            d1 = 2*h*d1./norm(d1);
        end

        d2 = (x0-y) - (dot(x0-y, g)/dot(g,g))*g;
        if norm(d2) > 2*h
            d2 = 2*h*d2./norm(d2);
        end
        
        % If the step over-shoots the target interpolate to the zero
        cellID2 = pointLocation(TR, y + d1);   
        rbfData = setRBF(cellID2, IC, rbfData);
        f = rbfData.hrbf{cellID2}.evaluateHRBF((y+d1)');                            
        if p*f(1) <= 0
            f = p/(p - f(1));
            d1 = min(f, 1)*d1;
        end
        
        c0 = dot(cross([g 0], [x0-y 0]), [0 0 1]);
        cellID2 = pointLocation(TR, y + d2);   
        rbfData = setRBF(cellID2, IC, rbfData);
        f = rbfData.hrbf{cellID2}.evaluateHRBF((y+d2)');                    
        c1 = dot(cross([f(2:end)' 0], [x0-y-d2 0]), [0 0 1]);
        if c0*c1 <= 0
            f = c0/(c0 - c1);
            d2 = min(f, 1)*d2;
        end     
        
        % In case the step leaves the domain.
        cellID = NaN;
        a = 2;
        while isnan(cellID)
            a = a/2;
            cellID = pointLocation(TR, y + a*(d1 + d2));                
        end            
        y = y + a*(d1 + d2);        
    end
end