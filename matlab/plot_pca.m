close('all');
clear('all'); 
%% library spm8 loading 
% addpath('C:\Users\ppsor\Desktop\Code\spm8');
% addpath ('C:\Users\ppsor\Desktop\Code\brain_image')
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/spm8/');
%% load data
% path = 'C:\Users\ppsor\Desktop\studi\Lionel\brains\';
path = '/home/kusch/Documents/project/patient_analyse//paper/result/PCA/';

paths = [
    ["/avalanches_pattern/vector_cluster.mat" "/avalanches_pattern/projection_"];
    ["/avalanches/vector_cluster.mat" "/avalanches/projection_"];
    ["/data_normalized/vector_cluster.mat" "/data_normalized/projection_"];
    ["/data/vector_cluster.mat" "/data/projection_"];
        ];
for path_data_save = paths.'
    load(strcat(path,path_data_save(1))); %% data for null model of pipeline
    
    nb_cluster = size(cluster_vector);
    nb_cluster = nb_cluster(1);
    data = squeeze(cluster_vector(:,:,:));
    data(data==0) = NaN;
    
    for kk1=1:nb_cluster
        plot_data(data(kk1,1:76), strcat(path,path_data_save(2),int2str(kk1-1),'.png'), true, ...
            [min(min(cluster_vector)), max(max(cluster_vector ))]);
    end
end