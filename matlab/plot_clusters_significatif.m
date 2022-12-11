close('all');
clear('all'); 
%% library spm8 loading 
addpath('C:\Users\ppsor\Desktop\Code\spm8');
addpath ('C:\Users\ppsor\Desktop\Code\brain_image')
% addpath('/home/kusch/Documents/project/patient_analyse/matlab/spm8');

%% load data
path = 'C:\Users\ppsor\Desktop\studi\Lionel\brains\';
%% load(strcat(path,'/paper/result/default/null_model/vector_null_trans.mat')); %% data for null model of pipeline
load(strcat(path,'\cluster_res_vector_null_trans.mat')); %% data for null model of region

nb_cluster = size(transmatrix);
nb_cluster = nb_cluster(1);
data = squeeze(transmatrix(:,:,:));
data(data==0) = NaN;

for kk1=1:nb_cluster
    plot_data(data(kk1,1:76), strcat(path,'\cluster_significatif_',int2str(kk1-1),'.png'), true, [1.0, 2.0]);
end