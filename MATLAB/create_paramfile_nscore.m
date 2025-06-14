function create_paramfile_nscore

%---------------------------------------------------
% Create a default parameter file for program nscore
%---------------------------------------------------
%
% Author: Xavier Emery

fid = fopen('nscore.par','w');

fprintf(fid,'%1s\n','                  Parameters for NSCORE');
fprintf(fid,'%1s\n','                  *********************');
fprintf(fid,'%1s\n',' ');
fprintf(fid,'%1s\n','START OF PARAMETERS:');
fprintf(fid,'%1s\n','data.dat                         % input file name');
fprintf(fid,'%1s\n','4:18                             %     column(s) for data values');
fprintf(fid,'%1s\n','-1.0 1.0e21                      % trimming limits for data values');
fprintf(fid,'%1s\n','nscore.out                       % output file with normal scores transforms');
fprintf(fid,'%1s\n','3                                % number of decimals for values in the output files');

fclose(fid);
