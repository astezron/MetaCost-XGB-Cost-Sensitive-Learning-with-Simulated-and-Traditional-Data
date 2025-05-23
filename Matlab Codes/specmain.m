function simu = specmain(coord,W,sigma,nvar,nlines,nrealiz,all_lines,all_r,all_phi)

%------------------------------------------------------------
% Non conditional spectral simulation
%------------------------------------------------------------
%
% Author: Xavier Emery

m = size(coord,1);
simu = zeros(m,nvar,nrealiz);

% Loop over the realizations

for k = 1:nrealiz

  % Project the points to simulate over the lines of the i-th nested structure
  index = [(k-1)*nlines*nvar+1:k*nlines*nvar]';
  lines = all_lines(index,:);
  x = coord*lines';

  % Simulate the values by a continuous spectral method
  r = all_r(index) * ones(1,m);
  phi = all_phi(index) * ones(1,m);
  simu(:,:,k) = sigma*cos(x.*r'+phi')*W(:,index)';

end

simu = reshape(simu,m,nvar*nrealiz);
