function [I,datacoord,datavalues] = searchdata(datacoord,datavalues,coord,search_rotationmatrix,octant,ndata);

%-------------------------------------------------------------------------------------------
% Search for the data located near location "coord" according to the neighborhood parameters
% (search_rotationmatrix,octant,ndata)
%-------------------------------------------------------------------------------------------

n = size(datacoord,1);
if n==0, I = zeros(0,1); return; end
I1 = [1:n]';


% Remove NaN data
%----------------

I0 = find(isnan(min(datavalues,[],2)));
if ~isempty(I0)
  datacoord(I0,:) = [];
  datavalues(I0,:) = [];
  I1(I0) = [];
end
n = size(datacoord,1);
if (n == 0), I = zeros(0,1); return; end


% Selection of data according to radius, angles and octant
%---------------------------------------------------------

% Compute the reduced distances between data and target location

deltacoord = datacoord-ones(n,1)*coord;
deltacoord = (deltacoord*search_rotationmatrix)';

% Create flags to indicate the belonging to angular sectors

if (octant == 0)
  nsector = 1;
  flag(1,:) = [1:n];
else
  nsector = 8;
  flag(1,:) = (deltacoord(1,:) > 0) & (deltacoord(2,:) > 0) & (deltacoord(3,:) > 0);
  flag(2,:) = (deltacoord(1,:) > 0) & (deltacoord(2,:) > 0) & (deltacoord(3,:) <= 0);
  flag(3,:) = (deltacoord(1,:) > 0) & (deltacoord(2,:) <= 0) & (deltacoord(3,:) > 0);
  flag(4,:) = (deltacoord(1,:) > 0) & (deltacoord(2,:) <= 0) & (deltacoord(3,:) <= 0);
  flag(5,:) = (deltacoord(1,:) <= 0) & (deltacoord(2,:) > 0) & (deltacoord(3,:) > 0);
  flag(6,:) = (deltacoord(1,:) <= 0) & (deltacoord(2,:) > 0) & (deltacoord(3,:) <= 0);
  flag(7,:) = (deltacoord(1,:) <= 0) & (deltacoord(2,:) <= 0) & (deltacoord(3,:) > 0);
  flag(8,:) = (deltacoord(1,:) <= 0) & (deltacoord(2,:) <= 0) & (deltacoord(3,:) <= 0);
end

% Select the neighboring data

I = zeros(ndata*nsector,1);
k = 0;

for i = 1:nsector

  index = find(flag(i,:) > 0);

  if ~isempty(index)

    squareddist = sum(deltacoord(:,index).^2);

    % Discard the data located beyond the radius
    J1 = find( squareddist<1 );

    % Sort the data by increasing distance
    [sorteddist,J] = sort(squareddist(J1));
    index = index(J1(J));
    n = min(ndata,length(J));
    index = index(1:n);
    I(k+1:k+n,1) = index(:);
    k = k+n;

  end

end

I = I(1:k,1);
datacoord = datacoord(I,:);
datavalues = datavalues(I,:);
I = I1(I);
