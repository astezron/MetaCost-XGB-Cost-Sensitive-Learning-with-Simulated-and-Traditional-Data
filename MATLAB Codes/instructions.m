%-------------------------------------------------------------------------------------------
% Main program files:
%    1. nscore: Gaussian transformation
%    2. gamv: calculation of experimental variograms
%    3. vargfit: semi-automatic fitting of a linear model of coregionalization
%    4. specsim: spectral simulation with nugget effect filtering
%
% Subroutines:
%    all the other *.m files
%
%-------------------------------------------------------------------------------------------

%--------------------------------------------
% MATLAB INSTRUCTIONS TO SIMULATE THE PROXIES
%--------------------------------------------

%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Requirement (input csv data files):
%    alldata.csv: file with the training+testing data set, with header (first row) followed by 12 columns corresponding to 3 coordinates, 1 response variable (categorical), 1 index (training or testing data) and 7 feature variables
%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

% Simulate the proxies based on information of feature variables only, in order to train the XGBoost classifier on the training subset

nscore('nscore.par');    % transform features to Gaussian variables
gamv('gamv.par');        % compute experimental direct and cross-variograms of Gaussian data
vargfit('vargfit.par');  % fit a linear model of coregionalization
specsim('specsim.par');  % simulate proxies at training+testing data locations, conditioned to training+testing feature data values

