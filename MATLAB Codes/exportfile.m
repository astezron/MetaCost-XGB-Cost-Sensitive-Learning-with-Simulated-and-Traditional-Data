function fid = exportfile(filename,filetitle,nfield,header,filecontent,format,nbdecimal,fileopen,fileclose);

%------------
% Export file
%------------
%
% Author: Xavier Emery


warning('off','all');

if nargin<9, fileclose = 1; end
if nargin<8, fileopen = 1; end
if nargin<7, nbdecimal = 5; end
if nargin<6, format=[]; end

if fileopen == 1
  fid = fopen(filename,'w');
  fprintf(fid,'%1s\n',filetitle);
  fprintf(fid,'%1s\n',int2str(nfield));
  for i = 1:nfield(1)
    fprintf(fid,'%1s\n',header{i});
  end
elseif fileopen == 2
  fid = fopen(filename,'w');
  for i = 1:nfield(1)-1
    fprintf(fid,'%1s,',header{i});
  end
  fprintf(fid,'%1s\n',header{i+1});
else
  fid = filename;
end

if isempty(format)
  for i = 1:nfield(1), format = [format,'  %.',int2str(nbdecimal(1)),'f']; end
elseif format<0
  format = [];
  for i = 1:nfield(1)-1, format = [format,'  %.',int2str(nbdecimal(1)),'f,']; end
  format = [format,'  %.',int2str(nbdecimal(1)),'f'];
end

if ~isempty(filecontent)
  I = find(isnan(filecontent(:)));
  filecontent(I) = -99*ones(size(I));
  fprintf(fid,[format,'\n'],filecontent');
end

if fileclose > 0
  fclose(fid);
end
