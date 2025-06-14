function weights = cokrige_main(datacoord,coord,model,sill,sillnugget,model_rotationmatrix,indexmissing);

%------------------------------------------------------------
% Compute cokriging weights and variance at locations "coord"
%------------------------------------------------------------
%
% Author: Xavier Emery

%------------------------------------------------------------------------------------------------------

% Definition of parameters
%-------------------------

n = size(datacoord,1);  % number of data
p = size(coord,1);      % number of target locations
nst = size(model,1);    % number of nested structures
nvar = size(sill,1);    % number of variables


% Calculation of the left covariance matrix K and right covariance matrix K0
%---------------------------------------------------------------------------

indexpoints = ones(nvar,1)*[1:n+p];
indexvariables = [1:nvar]'*ones(1,n+p);
indexpoints(indexmissing) = [];
indexvariables(indexmissing) = [];

% Calculation of matrix of reduced rotated distances

x = [datacoord;coord]; % coordinates of data + target locations
C = eye(n+p);
k = C(indexpoints(:)',indexpoints(:)').*sillnugget(indexvariables(:),indexvariables(:));
for i = 1:nst
  R = model_rotationmatrix(:,:,i);
  h = x*R;
  h = h*h';
  h = sqrt(max(0,-2*h+diag(h)*ones(1,n+p)+ones(n+p,1)*diag(h)'));
  C = cova(model(i,1),h);
  k = k + C(indexpoints(:)',indexpoints(:)').*sill(indexvariables(:),indexvariables(:),i);
end
q = n*nvar-length(indexmissing);
k0 = k(1:q,q+1:q+p*nvar);            % right member of kriging system
k1 = k(q+1:q+p*nvar,q+1:q+p*nvar);   % prior covariance
k = k(1:q,1:q);                      % left member of kriging system


% Calculation of cokriging weights
%---------------------------------

if q==0 % no data
  weights = zeros(0,p*nvar);

else % simple cokriging
  weights = k\k0;

end
