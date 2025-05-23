function create_paramfile_gamv

%-------------------------------------------------
% Create a default parameter file for program gamv
%-------------------------------------------------
%
% Author: Xavier Emery

fid = fopen('gamv.par','w');

fprintf(fid,'%1s\n','                  Parameters for GAMV');
fprintf(fid,'%1s\n','                  *******************');
fprintf(fid,'%1s\n',' ');
fprintf(fid,'%1s\n','START OF PARAMETERS:');
fprintf(fid,'%1s\n','nscore_training.out                             % file with data');
fprintf(fid,'%1s\n','1 2 3                                           %        columns for coordinates');
fprintf(fid,'%1s\n','19:33                                           %        columns for data values');
fprintf(fid,'%1s\n','Cu Au Mo As Bn Cp Cc Cv En Py Pyr Mol Ga Sph TS %        variable names');
fprintf(fid,'%1s\n','-10.0 10.0                                      %        trimming limits');
fprintf(fid,'%1s\n','2                                               % number of directions');
fprintf(fid,'%1s\n','0.0  90.0  0.0  20.0  15.0  20  7.5             %      direction 1: azm,atol,dip,dtol,lag,nlag,lagtol');
fprintf(fid,'%1s\n','0.0  90.0 90.0  20.0  10.0  20  5.0             %      direction 2: azm,atol,dip,dtol,lag,nlag,lagtol');
fprintf(fid,'%1s\n','gamv                                            % basename for output file');

fclose(fid);
