close('all');
clear('all'); 
%% library spm8 loading 
%addpath('C:\Users\ppsor\Desktop\Code\spm8');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/spm8');

%% load data
path = '/home/kusch/Documents/project/patient_analyse/';
load(strcat(path,'/paper/result/default/null_model/vector_null_trans.mat'));

nb_cluster = size(transmatrix);
nb_cluster = nb_cluster(1);
data = squeeze(transmatrix(:,:,:));
data(data==0) = NaN;

for kk1=1:nb_cluster
    plot_data(data(kk1,1:76), strcat(path,'/paper/result/default/plot_brain/cluster_significatif_',int2str(kk1-1),'.png'), false, [1.0, 2.0]);
end
