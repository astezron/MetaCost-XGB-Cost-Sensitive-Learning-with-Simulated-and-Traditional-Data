function vargfit(paramfile);

%----------------------------------------
% Fit a linear model of coregionalization
%----------------------------------------
%
% Author: Xavier Emery
%
% INPUT OF PARAMFILE:
%   gam     : file with direct and cross variograms/covariances
%   model   : variogram/covariance model (nst * 7 matrix, where nst is the number of nested structures)
%             Each row refers to a nested structure and is codified as: [type, ranges, angles]
%             There are three ranges (along the rotated y, x and z axes) and three angles to define the coordinate
%             rotation (azimuth, dip and plunge), see Deutsch and Journel, 1992, p. 25
%             Available types:
%                 1: spherical
%                 2: exponential
%   weight  : 0=no weighting
%             1=weights proportional to number of pairs
%             2=weights inversely proportional to distance
%             3=weights proportional to number of pairs and inversely proportional to distance
%   basename: basename for output files
%   vname   : variable names

%-----------------------------------------------------------------------------------------------------------------------------------

% This program is an adaptation of the code described in the following paper:
%   Emery X, 2010. Iterative algorithms for fitting a linear model of coregionalization. Computers & Geosciences 36(9), 1150-1160.


% Define parameters that user can modify

warning('off','all');
epsilon = 1e-4;
tmax = 250;

%-----------------------------------------------------------------------------------------------------------------------------------

% Prompt for parameter file if no input is entered
%-------------------------------------------------

if (nargin < 1)
  disp('Which parameter file do you want to use?');
  paramfile = input('','s');
end


% Read from parameter file
%-------------------------

if isempty(paramfile)
  paramfile = 'vargfit.par';
end

fid = fopen(paramfile);

if (fid < 0)
  disp('ERROR - The parameter file does not exist,');
  disp('        Check for the file and try again');
  disp(' ');
  disp('        creating a blank parameter file');
  disp(' ');
  disp('Stop - Program terminated.');
  create_paramfile_vargfit;
  return;
end

% The parameter file does exist
fgets(fid); fgets(fid); fgets(fid); fgets(fid);

tline = fgets(fid);
i = (find(tline == ' '));
if ~isempty(i), tline = tline(1:i(1)-1); end
gam = tline;

tline = fgets(fid);
tline = str2num(tline);
nst = tline(1);

for i = 1:nst
  tline = fgets(fid);
  tline = str2num(tline);
  model(i,:) = tline([1 2 3 4 5 6 7]);
end

tline = fgets(fid);
nugget = str2num(tline);

tline = fgets(fid);
weight = str2num(tline);

tline = fgets(fid);
i = (find(tline == ' '));
if ~isempty(i), tline = tline(1:i(1)-1); end
basename = tline;

tline = fgets(fid);
vname = tline;

fclose(fid);


% Read input variogram file

fid = fopen(gam);
if (fid < 0)
  fclose('all');
  error('ERROR - The file with variogram values does not exist');
end
azm2 = zeros(1,0);
dip2 = zeros(1,0);
nlag2 = zeros(1,0);
tail2 = zeros(1,0);
head2 = zeros(1,0);
gam2 = zeros(0,3);
while ~feof(fid)
  tline = fgets(fid);
  i = (find(tline == '('));
  j = (find(tline == ')'));
  eval([tline(i+1:j-1),';']);
  azm2 = [azm2 azm];
  dip2 = [dip2 dip];
  nlag2 = [nlag2 nlag];
  tail2 = [tail2 tail];
  head2 = [head2 head];
  for k = 1:nlag
    tline = fgets(fid);
    tline = str2num(tline);
    gam2 = [gam2;tline([2 4 3])];
  end
end
fclose(fid);
azm = azm2;
dip = dip2;
nlag = nlag2;
tail = tail2;
head = head2;
gam = gam2;


% Define parameters

azm = pi/180*azm(:);
dip = pi/180*dip(:);
nvariog = length(azm);
u = [sin(azm).*cos(dip) cos(azm).*cos(dip) sin(dip)]; % directing vector
nlag = [0;cumsum(nlag(:))];
n = nlag(nvariog+1);
nvar = max(tail);
if nugget>0
  model = [model(1,:);model];
  model(1) = 0;
end
nst = size(model,1);
cc = zeros(nst,nvar*nvar);
e = 10*epsilon;
if isempty(vname), vname = num2str([1:nvar]); end
for j = 1:nvar
  while vname(1)==' ', vname(1) = []; end
  i = find(vname==' ');
  names{j} = vname(1:i(1)-1);
  vname(1:i(1)-1) = [];
end


% Check NaN values and variograms at zero distance

for l = 1:nvariog
  I = find(isnan(gam(nlag(l)+1:nlag(l+1),3)));
  if ~isempty(I)
    gam(nlag(l)+I,:) = [];
    nlag(l+1:nvariog+1) = nlag(l+1:nvariog+1)-length(I);
  end
  I = find((abs(gam(nlag(l)+1:nlag(l+1),1))<eps)&(abs(gam(nlag(l)+1:nlag(l+1),3))<eps));
  if ~isempty(I)
    gam(nlag(l)+I,:) = [];
    nlag(l+1:nvariog+1) = nlag(l+1:nvariog+1)-length(I);
  end
end
n = nlag(nvariog+1);


% Define weights for least squares optimization

if weight == 0
  weights = ones(n,1);
elseif weight == 1
  weights = gam(:,2);
elseif weight == 2
  weights = 1./(1e-6+abs(gam(:,1)));
else
  weights = gam(:,2)./(1e-6+abs(gam(:,1)));
end
weights = weights./sum(weights);


% Calculate basic structures

h = [0 0 0];
for l = 1:nvariog, h = [h;gam(nlag(l)+1:nlag(l+1),1)*u(l,:)]; end
for i = 1:nst
  R = setrot(model,i);
  ha = h*R;
  ha = sqrt(sum(ha'.^2))';
  C = cova(model(i,1),ha)+0;
  g(:,i) = C;
end
g = ones(n,1)*g(1,:) - g(2:n+1,:);


% Iteration

t = 0;
k = 0;
while e>epsilon && t<tmax
  cc2 = cc;
  k = k+1;
  [ignore,I] = sort(rand(nst,1));
  tic;
  for i = 1:nst
    cci = cc;
    cci(I(i),:) = zeros(1,nvar*nvar);
    gammai = g*cci;
    deltai = zeros(nvar,nvar);
    wi = 1e-10*ones(nvar,nvar);
    for l = 1:nvariog
      gammail = reshape(gammai(nlag(l)+1:nlag(l+1),:),nlag(l+1)-nlag(l),nvar,nvar);
      gammail = gammail(:,tail(l),head(l));
      deltai(tail(l),head(l)) = deltai(tail(l),head(l)) + sum(weights(nlag(l)+1:nlag(l+1),:).*g(nlag(l)+1:nlag(l+1),I(i)).*(gam(nlag(l)+1:nlag(l+1),3)-gammail));
      wi(tail(l),head(l)) = wi(tail(l),head(l)) + weights(nlag(l)+1:nlag(l+1),:)'*g(nlag(l)+1:nlag(l+1),I(i)).^2;
    end
    deltai = deltai./wi;
    [Q,L] = eig(deltai);
    L = real(L);
    Q = real(Q);
    L2 = max(0,L);
    B = Q*L2*Q';
    B = (B+B')/2;
    cc(I(i),:) = B(:)';
    e = norm(cc2 - cc);
    t = t+toc;
  end


  % Calculate WSS (weighted sum of squares of fitting errors)

  gamma = g*cc;
  delta = zeros(nvar,nvar);
  for l = 1:nvariog
    gammal = reshape(gamma(nlag(l)+1:nlag(l+1),:),nlag(l+1)-nlag(l),nvar,nvar);
    gammal = gammal(:,tail(l),head(l));
    delta(tail(l),head(l)) = delta(tail(l),head(l)) + weights(nlag(l)+1:nlag(l+1),:)'*(gam(nlag(l)+1:nlag(l+1),3)-gammal).^2;
  end
  WSS = sum(delta(:));


  % Print results in output file

  format1 = [];
  for i = 1:nvar.^2
    format1 = [format1, ' %15.10f'];
  end
  format1 = [format1,'\n'];
  format2 = '%4.0f %15.3f %15.3f %15.3f %10.3f %10.3f %10.3f\n';
  model = min(model,1e10);
  if nugget > 0
    fid = fopen([basename,'.nug'],'w');
    fprintf(fid,format1,cc(1,:));
    fclose('all');
    fid = fopen([basename,'.mod'],'w');
    for i = 2:nst
      fprintf(fid,format2,model(i,:));
    end
    fclose('all');
    fid = fopen([basename,'.cc'],'w');
    for i = 2:nst
      fprintf(fid,format1,cc(i,:));
    end
    fclose('all');
  else
    fid = fopen([basename,'.nug'],'w');
    fprintf(fid,format1,0*cc(1,:));
    fclose('all');
    fid = fopen([basename,'.mod'],'w');
    for i = 1:nst
      fprintf(fid,format2,model(i,:));
    end
    fclose('all');
    fid = fopen([basename,'.cc'],'w');
    for i = 1:nst
      fprintf(fid,format1,cc(i,:));
    end
    fclose('all');
  end

end


% Display variogram fitting?
%---------------------------

h=zeros(0,3);
neg = 0;
for l = 1:nvariog
  hmax = max(gam(nlag(l)+1:nlag(l+1),1));
  if isempty(hmax), hmax = 0; end
  h = [h;[0:200]'*hmax/200*u(l,:)];
end
g = zeros(201*nvariog,nst);
for i = 1:nst
  R = setrot(model,i);
  ha = h*R;
  ha = sqrt(sum(ha'.^2))';
  C = cova(model(i,1),ha)+0;
  g(:,i) = C;
end
g = ones(size(g,1),1)*g(1,:) - g(1:size(g,1),:);
h = sqrt(sum(h.^2,2));
colors = [];
symbol = [];
while(size(colors,1)<nvariog), colors = [colors;['k';'b';'g';'r';'c';'m';'y']]; end
while(size(symbol,1)<nvariog), symbol = [symbol;['x';'+';'*';'o';'s';'d';'v';'^';'<';'>';'p';'h']]; end
count = zeros(nvar,nvar);
hmin = zeros(nvar,nvar);
hmax = zeros(nvar,nvar);
if min(gam(:,1))<0, neg = 1; end
for k = 1:nvar*nvar
  string{k} = [];
  j = floor((k-1)/nvar)+1;
  i = k-(j-1)*nvar;
  index = find((tail(1:nvariog)==i)&(head(1:nvariog)==j));
  if isempty(index), continue; end
  gamma = g*cc(:,k);
  figure(k);
  set(gcf,'DefaultAxesFontName','Times','DefaultAxesFontSize',14)
  set(gca,'Box','off');
  hold on;
  xlabel('Distance (m)');
  if i==j
    ylabel('Direct variogram');
    title(names{i});
  else
    ylabel('Cross-variogram');
    title([names{i},' & ',names{j}]);
  end
  for l = 1:length(index)
    count(i,j) = count(i,j)+1;
    plot(h((index(l)-1)*201+2:index(l)*201),gamma((index(l)-1)*201+2:index(l)*201),[colors(count(i,j),:) '-'],'LineWidth',1);
    string{k} = [string{k},'''azm = ',num2str(azm(index(l))*180/pi),'; dip = ',num2str(dip(index(l))*180/pi),''','];
  end
  eval(['hlegend = legend(',string{k},'''Location'',''SouthEast'',''AutoUpdate'',''off'');']);
  count(i,j) = 0;
  for l = 1:length(index)
    count(i,j) = count(i,j)+1;
    plot(gam(nlag(index(l))+1:nlag(index(l)+1),1),gam(nlag(index(l))+1:nlag(index(l)+1),3),[colors(count(i,j),:) symbol(count(i,j),:)]);
    if neg==1
      plot(-h((index(l)-1)*201+2:index(l)*201),gamma((index(l)-1)*201+2:index(l)*201),[colors(count(i,j),:) '-'],'LineWidth',1);
    end
    hmin(i,j) = min(hmin(i,j),min([0;gam(nlag(index(l))+1:nlag(index(l)+1),1)]));
    hmax(i,j) = max(hmax(i,j),max([h(index(l)*201);gam(nlag(index(l))+1:nlag(index(l)+1),1)]));
    hmax(i,j) = max(hmax(i,j),hmin(i,j)+1e-10);
    a = axis;
    if a(3)>0, a(3)=0; end
    if a(4)<0, a(4)=0; end
    axis([hmin(i,j) hmax(i,j) a(3) a(4)]);
  end
  if i==j
    print('-dpng',[basename,'_',names{i}],'-r300');
  else
    print('-dpng',[basename,'_',names{i},'_',names{j}],'-r300');
  end
  close all
end

close all

disp(' ');
disp('... Done...');
