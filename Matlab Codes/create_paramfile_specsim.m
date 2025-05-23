
function create_paramfile_specsim

%----------------------------------------------------
% Create a default parameter file for program specsim
%----------------------------------------------------
%
% Author: Xavier Emery

fid = fopen('specsim.par','w');

fprintf(fid,'%1s\n','                  Parameters for SPECSIM');
fprintf(fid,'%1s\n','                  **********************');
fprintf(fid,'%1s\n',' ');
fprintf(fid,'%1s\n','START OF PARAMETERS:');
fprintf(fid,'%1s\n','nscore_training.out              % file with coordinates of locations for co-simulation');
fprintf(fid,'%1s\n','1 2 3 19                         %        columns for location coordinates and response variable');
fprintf(fid,'%1s\n','nscore_training.out              % file with conditioning data');
fprintf(fid,'%1s\n','1 2 3                            %        columns for coordinates');
fprintf(fid,'%1s\n','19:33                            %        columns for Gaussian data');
fprintf(fid,'%1s\n','Cu Au Mo As Bn Cp Cc Cv En Py Pyr Mol Ga Sph TS  %        original variable names');
fprintf(fid,'%1s\n','-10.0  10.0                      %        trimming limits for Gaussian data');
fprintf(fid,'%1s\n','vargfit                          % basename for files with variogram models');
fprintf(fid,'%1s\n','100 100 60                       % search neighborhood: maximum radii in the rotated system');
fprintf(fid,'%1s\n','0 0 0                            %        angles for search ellipsoid');
fprintf(fid,'%1s\n','0                                %        divide into octants? 1=yes, 0=no');
fprintf(fid,'%1s\n','30                               %        optimal number of data per octant (if octant=1) or in total (if 0)');
fprintf(fid,'%1s\n','50                               % number of realizations');
fprintf(fid,'%1s\n','1000                             % number of turning lines');
fprintf(fid,'%1s\n','5000                             % maximum number of locations to simulate simultaneously');
fprintf(fid,'%1s\n','9784498                          % seed for random number generation');
fprintf(fid,'%1s\n','proxies_training.out             % name of output file');
fprintf(fid,'%1s\n','3                                % number of decimals for values in the output file');
fprintf(fid,'%1s\n',' ');

fclose(fid);
