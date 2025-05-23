function specsim

%-------------------------------------------------------------------------------------
% Spectral simulation of cross-correlated Gaussian random fields with nugget filtering
%-------------------------------------------------------------------------------------
%
% Author: Xavier Emery
%
% USE: specsim(paramfile);
%
% INPUT OF PARAMFILE:
%
%   LOCATIONS FOR CO-SIMULATION
%   ---------------------------
%   simucoord    : coordinates of the locations targeted for simulation (N * 3 matrix)
%
%   CONDITIONING DATA
%   -----------------
%   datacoord    : data coordinates (n * 3 matrix; void for non-conditional simulations)
%   datavalues   : conditioning Gaussian data (n * nvar matrix; void for non-conditional simulations)
%   vname        : original variable names
%   limits       : trimming limits (inf and sup) for the Gaussian data (1 * 2 vector)
%
%   COREGIONALIZATION MODEL
%   -----------------------
%   model        : covariance model for the Gaussian random fields (nst * 7 matrix, where nst is the number of nested structures)
%                  Each row refers to a nested structure and is codified as: [type, ranges, angles]
%                  There are three ranges (along the rotated y, x and z axes) and three angles to define the coordinate rotation
%                  (azimuth, dip and plunge), see Deutsch and Journel, 1992, p. 25
%                  Available types:
%                    1: spherical
%                    2: exponential
%   cc           : sills of the nested structures (nst * nvar^2 matrix)
%   nugget       : nugget effect variance-covariance matrix (1 * nvar^2 vector)
%
%   COKRIGING NEIGHBORHOOD
%   ----------------------
%   radius       : maximum search radii along y, x and z (rotated system) for conditioning data
%   angles       : angles for anisotropic search, according to GSLIB conventions (Deutsch and Journel, 1992, p. 25)
%   octant       : divide the neighborhood into octants? 1=yes, 0=no
%   ndata        : number of conditioning data per octant (if octant=1) or in total (if octant=0)
%
%   SIMULATION PARAMETERS
%   ---------------------
%   nrealiz      : number of realizations to draw
%   nlines       : number of turning lines to use for each nested structure
%   ntok         : number of target points to simulate simultaneously before exporting to output file
%   seed         : seed number for generating random values
%
%   OUTPUT OPTIONS
%   --------------
%   filename     : name of output file
%   nbdecimal    : number of decimals for the values in output file

%-----------------------------------------------------------------------------------------------------------------------------------

% This program is an adaptation of the algorithm described in the following papers:
%    Adeli A, Emery X, 2021. Geostatistical simulation of rock physical and geochemical properties with spatial filtering and its application to predictive geological mapping. Journal of Geochemical Exploration 220, article 106661.
%    Emery X, Arroyo D, Porcu E, 2016. An improved spectral turning-bands algorithm for simulating stationary vector Gaussian random fields. Stochastic Environmental Research and Risk Assessment 30(7), 1863-1873.
%
% It uses the following subroutines:
%   cova.m                     : compute covariance values
%   create_paramfile_specsim.m : create default parameter file
%   cokrige_main.m             : compute co-kriging weights
%   exportfile.m               : print output file
%   gamrndx.m                  : generate gamma distributed random values
%   importfile.m               : read data from GSLIB file
%   searchdata.m               : search the indices of the data located in the neighborhood
%   setrot.m                   : set up matrix for rotation and reduction of coordinates
%   specmain.m                 : main routine for simulation along the lines
%   specdens.m                 : compute spectral density of a covariance model
%   vdc.m                      : create equidistributed directions on the sphere (Van der Corput sequence)

%-----------------------------------------------------------------------------------------------------------

warning('off','all');

% Prompt for parameter file if no input is entered
%-------------------------------------------------

if (nargin < 1)
  disp('Which parameter file do you want to use?');
  paramfile = input('','s');
end


% Read from parameter file
%-------------------------

if isempty(paramfile)
  paramfile = 'specsim.par';
end

fid = fopen(paramfile);

if (fid < 0)
  disp('ERROR - The parameter file does not exist,');
  disp('        Check for the file and try again');
  disp(' ');
  disp('        creating a blank parameter file');
  disp(' ');
  disp('Stop - Program terminated.');
  create_paramfile_specsim;
  return;
else
  disp(' ');
  disp('Reading parameter file');
end

% The parameter file does exist
fgets(fid); fgets(fid); fgets(fid); fgets(fid);

tline0 = fgets(fid);
i = (find(tline0 == ' '));
if ~isempty(i), tline0 = tline0(1:i-1); end

tline = fgets(fid);
index0 = str2num(tline);

[filetitle,nfield,header0,simucoord] = importfile(tline0,[],1,index0);

tline0 = fgets(fid);
i = (find(tline0 == ' '));
if ~isempty(i), tline0 = tline0(1:i(1)-1); end

tline = fgets(fid);
index1 = str2num(tline);

tline = fgets(fid);
index2 = str2num(tline);
nvar = length(index2);

[filetitle,nfield,header,datacoord,datavalues] = importfile(tline0,[],1,index1,index2);

tline = fgets(fid);
vname = tline;

tline = fgets(fid);
limits = str2num(tline);

tline = fgets(fid);
j = find((tline == ' ')|(tline == char(10))|(tline == char(13)));
if ~isempty(j), tline = tline(1:j(1)-1); end
fid3 = fopen([tline,'.mod']);
if (fid3 > -1), model = load([tline,'.mod'],'-ascii'); fclose(fid3); else, error(['Unable to read file ',tline,'.mod']); end
fid3 = fopen([tline,'.cc']);
if (fid3 > -1), cc = load([tline,'.cc'],'-ascii'); fclose(fid3); else, error(['Unable to read file ',tline,'.cc']); end
fid3 = fopen([tline,'.nug']);
if (fid3 > -1), nugget = load([tline,'.nug'],'-ascii'); fclose(fid3); else, error(['Unable to read file ',tline,'.nug']); end

tline = fgets(fid);
radius = str2num(tline);

tline = fgets(fid);
angles = str2num(tline);

tline = fgets(fid);
octant = str2num(tline);

tline = fgets(fid);
ndata = str2num(tline);

tline = fgets(fid);
nrealiz = str2num(tline);

tline = fgets(fid);
nlines = str2num(tline);

tline = fgets(fid);
ntok = str2num(tline);

tline = fgets(fid);
seed = str2num(tline);

filename = fgets(fid);
i = (find(filename == ' '));
if ~isempty(i), filename = filename(1:i(1)-1); end

tline = fgets(fid);
nbdecimal = str2num(tline);

fclose(fid);

disp(' ');
disp('Preparing simulation');

% Define default values
%----------------------

nst = size(model,1);     % number of nested structures
nvar = sqrt(size(cc,2)); % number of variables
if nvar > floor(nvar), error('The number of columns in the sill matrix (cc) is inconsistent'); end
if length(radius) ~= 3, radius = radius(1)*ones(1,3); end
nrealiz = nvar*nrealiz;  % number of realizations
for j = 1:nvar
  while vname(1)==' ', vname(1) = []; end
  i = find(vname==' ');
  names{j} = vname(1:i(1)-1);
  vname(1:i(1)-1) = [];
end
minsimucoord = min(simucoord(:,1:length(index1)),[],1);
maxsimucoord = max(simucoord(:,1:length(index1)),[],1);
delta = maxsimucoord-minsimucoord;

% Simulation parameters
a = norm(delta(1:3))/64;  % frequencies of cosine waves
b = 0.3;
nlines = nlines(1);

% Model parameters
model(:,2:4) = min(model(:,2:4),10*norm(delta(1:3)));  % fix excessive ranges (zonal anisotropies)
sill = zeros(nvar,nvar,nst);
for i = 1:nst
  sill(:,:,i) = reshape(cc(i,:),nvar,nvar);
  R = setrot(model,i);
  model_rotationmatrix(:,:,i) = R;
  if max(abs(sill(:,:,i)-sill(:,:,i)'))>100*eps, error(['The sill matrix for structure nº',num2str(i),' is not symmetric']); end
  [eigenvectors,eigenvalues] = eig(sill(:,:,i));
  eigenvalues = max(0,eigenvalues);
  sill(:,:,i) = eigenvectors*eigenvalues*eigenvectors';
  if min(diag(eigenvalues))<0, error(['The sill matrix for structure nº',num2str(i),' is not positive semi-definite']); end
end

sillnugget = reshape(nugget,nvar,nvar);
if max(abs(sillnugget-sillnugget'))>100*eps, error(['The sill matrix for the nugget effect is not symmetric']); end
[eigenvectors,eigenvalues] = eig(sillnugget);
A0 = sqrt(eigenvalues)*eigenvectors';
if min(diag(eigenvalues))<0, error(['The sill matrix for the nugget effect is not positive semi-definite']); end
max_nugget = max(abs(nugget));


% Set data whose values are not in the trimming limits interval to NaN
%---------------------------------------------------------------------

m0 = size(datacoord,1);
if m0 > 0
  I = find( (datavalues(:) < limits(1)) | (datavalues(:) > limits(2)) );
  datavalues(I) = NaN*ones(size(I));
end


% Remove data located too far from the locations to simulate
%-----------------------------------------------------------

m0 = size(datacoord,1);
search_rotationmatrix = setrot([1 radius angles],1); % rotation-reduction matrix for data search

if (m0 > 0)

  % Reduced-rotated data coordinates
  tmp = datacoord*search_rotationmatrix;

  % Extremal points to simulate
  x = [minsimucoord(1) minsimucoord(2) minsimucoord(3); ...
       minsimucoord(1) minsimucoord(2) maxsimucoord(3); ...
       minsimucoord(1) maxsimucoord(2) minsimucoord(3); ...
       minsimucoord(1) maxsimucoord(2) maxsimucoord(3); ...
       maxsimucoord(1) minsimucoord(2) minsimucoord(3); ...
       maxsimucoord(1) minsimucoord(2) maxsimucoord(3); ...
       maxsimucoord(1) maxsimucoord(2) minsimucoord(3); ...
       maxsimucoord(1) maxsimucoord(2) maxsimucoord(3)];
  x = x*search_rotationmatrix;
  minx = min(x(:,1));
  miny = min(x(:,2));
  minz = min(x(:,3));
  maxx = max(x(:,1));
  maxy = max(x(:,2));
  maxz = max(x(:,3));

  % Identify and remove the data located beyond the search radii
  I = find( (tmp(:,1) < minx-1) | (tmp(:,1) > maxx+1) | ...
            (tmp(:,2) < miny-1) | (tmp(:,2) > maxy+1) | ...
            (tmp(:,3) < minz-1) | (tmp(:,3) > maxz+1) );
  datacoord(I,:) = [];
  datavalues(I,:) = [];
  m0 = size(datacoord,1);

end


% Create seed numbers
%--------------------

rand('state',seed);
randn('state',seed);
seed_vdc = ceil(1e7*rand);
seed_gam = ceil(1e7*rand);


%--------------------------------------------------------------------------------------------

% PREPARE THE LINES
%------------------

% Initialization

all_lines = vdc(nlines,nrealiz,seed_vdc);
sigma = sqrt(1./nlines);
G = gamrndx(b,nlines*nrealiz,seed_gam);
G = max(G,1e-10);
t = randn(nlines,nvar,nrealiz/nvar).^2 + randn(nlines,nvar,nrealiz/nvar).^2 + randn(nlines,nvar,nrealiz/nvar).^2;
all_r = sqrt(t)./reshape(sqrt(G*2),nlines,nvar,nrealiz/nvar)/a;
all_phi = 2*pi*rand(nlines,nvar,nrealiz/nvar);

% Importance sampling

W = zeros(nvar,nlines,nvar,nrealiz/nvar);
for l = 1:nlines*nrealiz
  i = floor((l-1)/nlines);
  iline = l-i*nlines;
  irealiz = floor(i/nvar);
  ivar = i-irealiz*nvar;
  u = all_r(iline,ivar+1,irealiz+1)*all_lines(l,:);
  g = a.^3*gamma(b+1.5)/gamma(b)/(pi.^1.5)./(1+a.^2*all_r(iline,ivar+1,irealiz+1).^2).^(b+1.5);
  f = 0;
  for k = 1:nst
    if model(k,1)==2
      R = setrot(model(k,:)./[1 3 3 3 1 1 1],1);
    else
      R = setrot(model,k);
    end
    R = inv(R);
    v = u*R';
    f0 = specdens(model(k,1),v,sill(:,:,k),det(R));
    f = f+f0;
  end
  H = 2*f./g;
  [V,D] = eig(H);
  temp = V*max(0,real(sqrt(D)));
  W(:,iline,ivar+1,irealiz+1) = temp(:,ivar+1);
end
W = reshape(W,nvar,nlines*nrealiz);

%--------------------------------------------------------------------------------------------

% NON CONDITIONAL SIMULATION AT DATA LOCATIONS
%---------------------------------------------

simudata = zeros(m0,nrealiz);
residuals = zeros(m0,nrealiz);

if (m0 > 0)

  disp(' ');
  disp('Non-conditional simulation at data locations');


  % How many data locations can be simulated simultaneously?
  %---------------------------------------------------------

  m2 = max(1,min(m0,ntok));
  sequence = [[0:m2:m0-0.5] m0];
  seed_nugget_data = ceil(1e7*rand);


  % Loop over the sequences of data points
  %---------------------------------------

  for n = 1:length(sequence)-1
    index = [sequence(n)+1:sequence(n+1)]';
    simudata(index,:) = specmain(datacoord(index,:),W,sigma,nvar,nlines,nrealiz/nvar,all_lines,all_r(:),all_phi(:));
  end
  simudata = real(simudata);


  % Add nugget effect
  %------------------

  if max_nugget > eps
    randn('state',seed_nugget_data);
    simunug = randn(m0*nrealiz/nvar,nvar)*A0;
    simunug = reshape(simunug',nrealiz,m0);
    simudata = simudata + simunug';
  end


  % Prepare conditioning co-kriging
  %--------------------------------

  residuals = kron(ones(1,nrealiz/nvar),datavalues)-simudata;

end

%--------------------------------------------------------------------------------------------

% Open output file
%-----------------

filetitle = 'Simulated values';
for j = 1:length(index0)
  header2{j} = header0{index0(j)};
end
for j = 1:nvar
  header2{j+length(index0)} = ['Simulated ',names{j}];
end
fid = exportfile(filename,filetitle,nvar+length(index0),header2,[],[],nbdecimal(1),2,0);
progress = 0;


%--------------------------------------------------------------------------------------------

% CONDITIONAL SIMULATION AT TARGET LOCATIONS
%-------------------------------------------

disp(' ');
disp('Simulation at target locations');


% How many locations can be simulated simultaneously?
%----------------------------------------------------

m1 = size(simucoord,1);
m2 = max(1,min(m1,ntok));
sequence  = [[0:m2:m1-0.5] m1];
lengthsequence = length(sequence)-1;
seed_nugget = ceil(1e7*rand(1,lengthsequence));


% Loop over the sequences of points to simulate
%----------------------------------------------

for ni = 1:lengthsequence*m2

  n = 1+floor((ni-1)/m2);
  i = ni-(n-1)*m2;

  if i == 1

    % Coordinates of the points to simulate
    index = [sequence(n)+1:sequence(n+1)]';
    m1 = length(index);
    if m0 == 0, m3 = -1; else, m3 = m1; end
    coord = simucoord(index,1:length(index1));

    % Non-conditional simulation
    simu = specmain(coord,W,sigma,nvar,nlines,nrealiz/nvar,all_lines,all_r(:),all_phi(:));
    simu = real(simu);

  end


  % Conditioning
  %-------------

  if i <= m3

    % Search for neighboring data
    [ignore,datacoord_i,residuals_i] = searchdata(datacoord,residuals,coord(i,1:3),search_rotationmatrix,octant,ndata);

    % Substitution of residuals
    n_i = size(datacoord_i,1);
    index_missing = find(isnan(residuals_i(:,1:nvar)'));
    weights = cokrige_main(datacoord_i,coord(i,:),model,sill,sillnugget+1e-7,model_rotationmatrix,index_missing);
    residuals_i = reshape(residuals_i,n_i,nvar,nrealiz/nvar);
    residuals_i = permute(residuals_i,[2 1 3]);
    residuals_i = reshape(residuals_i,n_i*nvar,nrealiz/nvar);
    residuals_i(index_missing,:) = [];
    epsilon = weights'*residuals_i;
    epsilon = reshape(epsilon,nvar,1,nrealiz/nvar);
    epsilon = permute(epsilon,[2 1 3]);
    epsilon = reshape(epsilon,1,nrealiz);
    simu(i,:) = simu(i,:) + epsilon;

  end

  if i == m1

    % Report on progress from time to time
    %-------------------------------------

    progress2 = 10*floor((10*n)/length(sequence));
    if (progress2 > progress)
      disp(['  ',num2str(progress2),'% completed']);
      progress = progress2;
      pause(0.001);
    end


    % Write in output file
    %---------------------

    simu2 = reshape(simu,m1,nvar,nrealiz/nvar);
    simu2 = permute(simu2,[3,1,2]);
    simu2 = reshape(simu2,nrealiz/nvar*m1,nvar);
    exportfile(fid,filetitle,nvar+length(index0),header2,[kron(simucoord,ones(nrealiz/nvar,1)) simu2],-1,nbdecimal(1),0,0);

  end

end


% CLOSE THE OUTPUT FILE
%----------------------

fclose(fid);
disp(' ')
disp('... Done...')
