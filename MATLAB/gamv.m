function gamv(paramfile);

%---------------------------------------------------
% Calculate experimental direct and cross variograms
%---------------------------------------------------
%
% Author: Xavier Emery
%
% USE: gamv(paramfile);
%
% INPUT OF PARAMFILE:
%   datacoord         : data coordinates
%   datavalues        : data values
%   vname             : variable names
%   limits            : trimming limits and top-cut values for data
%   azm, atol         : azimuth and azimuth tolerance (in degrees) for experimental variogram calculation
%   dip, dtol         : dip and dip tolerance (in degrees) for experimental variogram calculation
%   lag, nlag, lagtol : lag, number of lags and lag tolerance for experimental variogram calculation
%   basename          : basename for output files with experimental variograms

%-----------------------------------------------------------------------------------------------------------

% User defined parameters
%------------------------

fontname = 'Times';
fontsize = 14;
allcolors = ['k';'b';'g';'r';'c';'m';'y'];   % colors for representing directions in variogram display
allsymbols = ['x';'+';'*';'o';'s';'d';'v'];  % symbols for representing directions in variogram display

%-----------------------------------------------------------------------------------------------------------

warning('off','all');


% Read from parameter file
%-------------------------

if (nargin < 1) % Prompt for parameter file
  disp('Which parameter file do you want to use?');
  paramfile = input('','s');
end

if isempty(paramfile)
  paramfile = 'gamv.par';
end
fid = fopen(paramfile);
if (fid < 0)
  disp('ERROR - The parameter file does not exist,');
  disp('        Check for the file and try again');
  disp(' ');
  disp('        creating a blank parameter file');
  disp(' ');
  disp('Stop - Program terminated.');
  create_paramfile_gamv;
  return;
end

% The parameter file does exist

fgets(fid); fgets(fid); fgets(fid); fgets(fid);

tline0 = fgets(fid);
i = (find(tline0 == ' '));
if ~isempty(i), tline0 = tline0(1:i(1)-1); end

tline = fgets(fid);
index1 = str2num(tline);

tline = fgets(fid);
index2 = str2num(tline);

[filetitle,nfield,header,datacoord,datavalues] = importfile(tline0,[],1,index1,index2);

tline = fgets(fid);
vname = tline;

tline = fgets(fid);
limits = str2num(tline);

tline = fgets(fid);
ndir = str2num(tline);

azm = NaN*ones(1,ndir);
atol = NaN*ones(1,ndir);
dip = NaN*ones(1,ndir);
dtol = NaN*ones(1,ndir);
lag = NaN*ones(1,ndir);
nlag = NaN*ones(1,ndir);
lagtol = NaN*ones(1,ndir);
for j = 1:ndir
  tline = fgets(fid);
  tline = str2num(tline);
  azm(j) = tline(1);
  atol(j) = tline(2);
  dip(j) = tline(3);
  dtol(j) = tline(4);
  lag(j) = tline(5);
  nlag(j) = tline(6);
  lagtol(j) = tline(7);
end

basename = fgets(fid);
i = (find(basename == ' '));
if ~isempty(i), basename = basename(1:i(1)-1); end

fclose(fid);


% Define parameters
%------------------

n = size(datacoord,1);
ndir = length(azm);
nvar = length(index2);


% Check input
%------------

I = find( (datavalues(:) < limits(1)) | (datavalues(:) > limits(2)) );
datavalues(I) = NaN*ones(size(I));


% Define names for output files and graphics
%-------------------------------------------

% Variable names
while vname(1)==' ', vname(1) = []; end
if (vname(1)=='%')||(vname(1)==char(10))||(vname(1)==char(13))
  for j = 1:nvar
    vname = header{index2(j)};
    i = find(vname==' ');
    for k = 1:length(i), vname(i(k))='_'; end
  end
end
for j = 1:nvar
  while vname(1)==' ', vname(1) = []; end
  i = find(vname==' ');
  if ~isempty(i)
    names{j} = vname(1:i(1)-1);
    vname(1:i(1)-1) = [];
  else
    names{j} = vname;
  end
end

% Legend location
location = 'SouthEast';


% Characteristics of lag separation vectors
%------------------------------------------

alllag = zeros(0,1);
allazm = zeros(0,1);
allatol = zeros(0,1);
alldip = zeros(0,1);
alldtol = zeros(0,1);
alllagtol = zeros(0,1);
for i = 1:ndir
  while dip(i)>90, dip(i) = dip(i)-180; azm(i) = azm(i)+180; end
  while dip(i)<-90, dip(i) = dip(i)+180; azm(i) = azm(i)+180; end
  alllag = [alllag;[0 lag(i)/4 lag(i):lag(i):nlag(i)*lag(i)]'];
  allazm = [allazm;azm(i)*ones(nlag(i)+2,1)];
  allatol = [allatol;atol(i)*ones(nlag(i)+2,1)];
  alldip = [alldip;dip(i)*ones(nlag(i)+2,1)];
  alldtol = [alldtol;dtol(i)*ones(nlag(i)+2,1)];
  alllagtol = [alllagtol;[1e-10;lagtol(i)/2-1e-10;lagtol(i)*ones(nlag(i),1)]];
end


% Calculate experimental variograms
%----------------------------------

disp(' ')
disp('Calculating experimental variograms')

% Initialization
npairs = zeros(nvar,nvar,length(alllag));
varexp = zeros(nvar,nvar,length(alllag));
dist = zeros(nvar,nvar,length(alllag));
progress = 0;


% Loop over data
for i = 1:n

  % Calculate azimuth, dip and lag separation between data i and data j
  h = ones(n,1)*datacoord(i,:)-datacoord;
  azmij = atan(h(:,1)./(1e-10+h(:,2)))*180/pi;
  azmij = azmij+180*(h(:,2)<0);
  dipij = atan(h(:,3)./sqrt(max(1e-10,h(:,1).^2+h(:,2).^2)))*180/pi;
  lagij = sqrt(h(:,1).^2+h(:,2).^2+h(:,3).^2);

  % Loop over lags
  for l = 1:length(alllag)

    % Find indices of data paired with data i

    if alllag(l)<1e-10
      J = find(lagij<1e-10);
      if isempty(J), continue; end
    else
      J1 = find(abs(alllag(l)-lagij)<=alllagtol(l));
      if isempty(J1), continue; end
      J2 = find(abs(alldip(l)-dipij(J1))<=alldtol(l));
      if isempty(J2), continue; end
      J3 = find(  (mod(allazm(l)-azmij(J1(J2)),360)<=allatol(l)) | ((360-mod(allazm(l)-azmij(J1(J2)),360))<=allatol(l)) ) ;
      if isempty(J3), continue; end
      J = J1(J2(J3));
    end

    % Store tail and head information
    lagvalues = lagij(J);
    values1 = ones(size(J))*datavalues(i,:);
    values2 = datavalues(J,:);

    % Loop on pairs of variables
    for k = 1:nvar*nvar
      i1 = floor((k-1)/nvar)+1;
      i2 = k-(i1-1)*nvar;
      temp = (values1(:,i1)-values2(:,i1)).*(values1(:,i2)-values2(:,i2));
      valid = find(~isnan(temp));
      l1 = lagvalues(valid,1);
      npairs(i1,i2,l) = npairs(i1,i2,l) + length(valid);
      varexp(i1,i2,l) = varexp(i1,i2,l) + sum(temp(valid,1));
      dist(i1,i2,l) = dist(i1,i2,l) + sum(l1);
    end

  end

  % Report on progress from time to time
  progress2 = 10*floor(10*i/n);
  if (progress2 > progress)
    disp(['  ',num2str(progress2),'% completed']);
    progress = progress2;
    pause(0.001);
  end

end

% Calculate final variogram values
dist = dist./npairs;
varexp = 0.5*varexp./npairs;


% Print results in output files
%------------------------------

disp('Printing results in output files');

clag = [0 cumsum(nlag+2)];
outputformat = ['%5.0f %12.3f %15.5f %12.0f'];
colors = [];
symbol = [];
while(size(colors,1)<ndir), colors = [colors;allcolors]; end
while(size(symbol,1)<ndir), symbol = [symbol;allsymbols]; end

fid = fopen([basename,'_variogram.out'],'w');

for i1 = 1:nvar
  for i2 = 1:nvar
    for j = 1:ndir
      g = varexp(i1,i2,clag(j)+1:clag(j+1));
      d = dist(i1,i2,clag(j)+1:clag(j+1));
      n = npairs(i1,i2,clag(j)+1:clag(j+1));
      valid = find(n(:)>0);
      d = d(:); g = g(:); n = n(:);
      fprintf(fid,'%1s\n',['Variogram for ',names{i1},' & ',names{i2},': Direction n°',int2str(j),' (azm = ',num2str(azm(j)),'; dip = ',num2str(dip(j)),'; nlag = ',num2str(length(valid)),'; tail = ',int2str(i1),'; head = ',int2str(i2),')']);
      fprintf(fid,[outputformat,'\n'],[[1:length(valid)]' d(valid) g(valid) n(valid)]');
    end
  end
end

fclose(fid);

disp(' ');
disp('... Done...');
