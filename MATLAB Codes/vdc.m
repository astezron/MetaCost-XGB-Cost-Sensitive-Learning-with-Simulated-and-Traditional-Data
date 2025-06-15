function lines = vdc(nlines,nrealiz,seed)

%------------------------------------------------------------
% Generation of equidistributed lines over the unit 3D sphere
% according to a low discrepancy (Van der Corput) sequence
%------------------------------------------------------------
%
% Author: Xavier Emery

lines = zeros(nlines*nrealiz,3);
rand('state',seed);
seed2 = ceil(1e7*rand);
randn('state',seed2);

i = [1:nlines]';

% binary decomposition of i
j = i;
u = 0;
p = 0;
while (max(j)>0)
  p = p+1;
  t = fix(j/2);
  u = u+2*(j/2-t)./(2.^p);
  j = t;
end

% ternary decomposition of i
j = i;
v = 0;
p = 0;
while (max(j)>0)
  p = p+1;
  t = fix(j/3);
  v = v+3*(j/3-t)./(3.^p);
  j = t;
end

% directing vector of the i-th line
x  = [cos(2*pi*u).*sqrt(1-v.*v) sin(2*pi*u).*sqrt(1-v.*v) v];

% random rotation
for k = 1:nrealiz
  angles = 360*rand(1,3);
  R = setrot([1 1 1 1 angles],1);
  lines((k-1)*nlines+1:k*nlines,:) = x*R;
end
