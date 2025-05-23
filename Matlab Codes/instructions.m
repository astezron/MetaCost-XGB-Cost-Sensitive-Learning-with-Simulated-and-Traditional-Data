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

%---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Requirement (input csv data files):
%    training.csv: file with the training data set, with header (first row) followed by 19 columns corresponding to 3 coordinates, 1 response variable (categorical) and 15 feature variables
%    alldata.csv: file with the training+testing data set, with header (first row) followed by 19 columns corresponding to 3 coordinates, 1 response variable (categorical) and 15 feature variables
%---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

% Simulate the proxies based on training data only, in order to train the XGBoost classifier

nscore('nscore1.par');   % transform features to Gaussian variables
gamv('gamv.par');        % compute experimental direct and cross-variograms of Gaussian data
vargfit('vargfit.par');  % fit a linear model of coregionalization
specsim('specsim1.par')  % simulate proxies at training data locations, conditioned to training data values

% Once the classifier is trained, simulate the proxies based on all the data (training + testing sets), in order to assess classification scores and classification uncertainty

nscore('nscore2.par');   % transform features to Gaussian variables
specsim('specsim2.par')  % simulate proxies at training+testing data locations, conditioned to training+testing data values

