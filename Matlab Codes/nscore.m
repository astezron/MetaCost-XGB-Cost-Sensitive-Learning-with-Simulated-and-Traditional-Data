function nscore(paramfile);

%-----------------------------
% Normal scores transformation
%-----------------------------
%
% Author: Xavier Emery
%
% INPUT IN PARAMFILE:
%   inputfilename : file with data values
%   index         :      column(s) for data values
%   limits        : trimming limits for data values
%   filename      : name for output file with normal scores transforms
%   nbdecimal     : number of decimals for the values in output file

%-----------------------------------------------------------------------------------------------

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
  paramfile = 'nscore.par';
end

fid = fopen(paramfile);

if (fid < 0)
  disp('ERROR - The parameter file does not exist,');
  disp('        Check for the file and try again');
  disp(' ');
  disp('        creating a blank parameter file');
  disp(' ');
  disp('Stop - Program terminated.');
  create_paramfile_nscore;
  return;
end

% The parameter file does exist
fgets(fid); fgets(fid); fgets(fid); fgets(fid);

tline = fgets(fid);
i = (find(tline == ' '));
if ~isempty(i), tline = tline(1:i(1)-1); end
inputfilename = tline;

tline = fgets(fid);
index = str2num(tline);

tline = fgets(fid);
limits = str2num(tline);

tline = fgets(fid);
i = (find(tline == ' '));
if ~isempty(i), tline = tline(1:i(1)-1); end
filename = tline;

tline = fgets(fid);
nbdecimal = str2num(tline);

fclose('all');


% Read input file
%----------------

[filetitle,nfield,allheader,allinput] = importfile(inputfilename,-1,1);


% Initialization
%---------------

zdata = allinput(:,index);
[n,nvar] = size(zdata);
ydata = -99*ones(n,nvar);


% Loop on variables
%------------------

for j = 1:nvar

  K = find( ~isnan(zdata(:,j)) & (zdata(:,j)<=limits(2)) & (zdata(:,j)>=limits(1)) );
  if isempty(K), continue; end
  weight = ones(length(K),1)/length(K);
  [data,I] = sort(zdata(K,j));
  cumulativeweight = cumsum(weight(I))-weight(I)/2;
  ydata(K(I),j) = norminv(cumulativeweight);
  allheader{nfield+j} = ['Normal scores for ',allheader{index(j)}];

end


% Write in output file
%---------------------

exportfile(filename,filetitle,nfield+nvar,allheader,[allinput ydata],[],nbdecimal,1,1);

disp(' ');
disp('... Done...')
