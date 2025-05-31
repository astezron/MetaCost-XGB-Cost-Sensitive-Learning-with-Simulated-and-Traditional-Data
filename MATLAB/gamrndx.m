function x = gamrndx(a,n,seed);

% Author: Xavier Emery
%
% Gamma variate generator, with shape parameter a<1
%--------------------------------------------------

b = 1+a*exp(-1);
c = 1/a;
rand('state',seed);
accept = 0;
x = NaN*zeros(n,1);

while (accept < 1)
  I = find(isnan(x));
  n = length(I);
  u = rand(n,1);
  v = b*u;
  w = rand(n,1);
  J = find(v<=1);
  x1 = v(J).^c;
  K = find(log(w(J))<=-x1);
  n1 = length(K);
  x(I(J(K))) = x1(K);
  J = find(v>1);
  x1 = -log(c*(b-v(J)));
  K = find(log(w(J))<=(a-1)*log(x1));
  n2 = length(K);
  x(I(J(K))) = x1(K);
  accept = (n-0.5<n1+n2)+0;
end
