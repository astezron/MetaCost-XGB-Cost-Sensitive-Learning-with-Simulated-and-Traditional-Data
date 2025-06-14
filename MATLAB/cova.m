function C = cova(it,h);

%------------------------------------------
% Compute covariance for reduced distance h
%------------------------------------------
%
% Author: Xavier Emery

warning('off','all');

if (it < 1) % Nugget effect

  C = (h<eps)+0;

elseif (it < 2) % Spherical model

  C = 1 - 1.5*min(h,1) + 0.5*(min(h,1).^3);

elseif (it < 3) % Exponential model

  C = exp(-3*h);

else

  error('Unavailable covariance model');

end
