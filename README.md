Standard/Modified Hermite Radial Basis Function Interpolation

This class implements both the standard Hermite RBF (HRBF) and Modified Hermite RBF (MHRBF)

Usage:
  * hrbf = HRBF(trainPoints, kernel, kernelParam, polyOrder, matrixType, modPow) % Create the HRBF/MHRBF object
  * hrbf = hrbf.computeWeights(@myFunction);  % Compute weights for the given function
  * After computing the weights of MHRBF/HRBF, there are two options: 
    * Evaluate function and derivative(s) at evaluation points
    * Compute interpolation error at given evaluation points
  * [InterpVals, evalMatrix] = hrbf.evaluateHRBF(evalPoints)  % Evaluate the function and derivatives at evaluation points
  * err = hrbf.computeError(evalPoints, actualValues, normType) % Computes interpolation error at given evaluation points   

Inputs:
  * trainPoints   - (d X n) array of n interpolation points in d-dimensional space
    * Used to construct the interpolation matrix for weight computation.
  * param         - Parameter for the radial basis function:
    * For 'GA' (Gaussian) and 'MQ' (Multiquadric), this is the shape parameter.
    * For 'PHS' (Polyharmonic Spline), this is degree of PHS kernel.
  * kernel        - Type of radial basis function:
    * 'GA'  (Gaussian)
    * 'MQ'  (Multiquadric)
    * 'PHS' (Polyharmonic Spline)
  * polyOrder     - Order of polynomial augmentation (-1 for none).
  * matrixType    - Specifies the interpolation matrix type:
    * 'HRBF'  (Standard Hermite RBF)
    * 'MHRBF' (Modified Hermite RBF)
  * modPow        - Exponent for MHRBF (only relevant for 'MHRBF'. Not required for 'HRBF'):
    * Represents the power term (n) in expressions like (x - x0)^n, (y - y0)^n.

Outputs:
  * hrbf          - HRBF object containing computed weights and methods.

Example:
  * hrbf = HRBF(trainPoints, 'GA', 1, 5, 'MHRBF', 4) 
  * hrbf = hrbf.computeWeights(@someFunction); % If training using a function handle
  * hrbf = hrbf.computeWeights(dataVals); % If training using a data values
  * [interpVals, evalMatrix] = hrbf.evaluateHRBF(evalPoints) % 
  * err = hrbf.computeError(evalPoints, actualValues, 'Linf')  
