function [h, ePhi, eH, eCP] = reinit_error(meshID)
%REINIT_ERROR Calculate the error for a given mesh
%   Input:
%       meshID - 1 is the coarsest mesh, 5 is the finest
%
%   Output:
%       h - Average distance between vertices
%       ePhi - 

    load('meshes.mat', 'meshes');

    if nargin==0
        meshID = 2;
    end
    pts = meshes{meshID}.pts;
    h = meshes{meshID}.h;
    shapeName = 'cassini';    
    phi0 = str2func(sprintf('reinit_%s', shapeName));
    shapeData = getfield(meshes{meshID}, shapeName);
    I = find(abs(shapeData.phi)<=0.1);
    % I = find(abs(shapeData.phi)<=5*h);

    rbfData.epsilon = 0.5;    
    rbfData.kernelPower = 4;
    rbfData.kernel = 'GA';
    rbfData.polyOrder = 1;
    rbfData.numHRBFPoints = 20; % Number of points to use for computing the HRBF
    rbfData.cp = shapeData.closestPoint(I, :);    
    rbfData.matType = 'MHRBF';
    rbfData.phi0 = phi0;
    rbfData.h = h;
        
    [phi, cp, H] = reinit(pts(I, :), pts, rbfData);

    e = (phi - shapeData.phi(I))./shapeData.phi(I);
    ePhi.linf = norm(e, 'inf');
    ePhi.l2 = norm(e, 2)/length(I);

    e = (H - shapeData.curv(I))./shapeData.curv(I);
    eH.linf = norm(e, 'inf');
    eH.l2 = norm(e, 2)/length(I);

    e = sqrt(sum((cp - shapeData.closestPoint(I,:)).^2, 2));
    eCP.linf = norm(e, 'inf');
    eCP.l2 = norm(e, 2)/length(I);
end