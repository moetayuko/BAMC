function setup_path()
cur_dir = fileparts(mfilename('fullpath'));
addpath(cur_dir);

files = dir(cur_dir);
dir_flags = [files.isdir];
sub_folders = files(dir_flags);
addpath(sub_folders(3:end).name);
