function F = specdens(it,u,C,a);

% Author: Xavier Emery
%
% Spectral densities of common covariance models
%-----------------------------------------------

warning('off','all');

if (it < 2) % Spherical model

   F = (3*a*C/(4*pi*(norm(u)).^3))*besselj(1.5,norm(u)/2).^2;

elseif (it < 3) % Exponential model

   F = (a*C)/pi/pi./(1+(norm(u)).^2).^2;

end
