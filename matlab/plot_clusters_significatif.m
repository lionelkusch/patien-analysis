close('all');
clear('all'); 
%% library spm8 loading 
% addpath('C:\Users\ppsor\Desktop\Code\spm8');
% addpath ('C:\Users\ppsor\Desktop\Code\brain_image')
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/spm8/');
%% load data
% path = 'C:\Users\ppsor\Desktop\studi\Lionel\brains\';
path = '/home/kusch/Documents/project/patient_analyse//paper/result/default/';

paths = [["/null_model/vector_null_trans.mat" "/null_model/figure/vector_null_trans_"];
    ["/null_model/0_2_vector_null_trans.mat" "/null_model/figure/0_2_vector_null_trans_"];
    ["/cluster_res_vector_null_trans.mat" "/figure/cluster_significatif_"]];
for path_data_save = paths.'
    load(strcat(path,path_data_save(1))); %% data for null model of pipeline
    
    nb_cluster = size(transmatrix);
    nb_cluster = nb_cluster(1);
    data = squeeze(transmatrix(:,:,:));
    data(data==0) = NaN;
    
    for kk1=1:nb_cluster
        plot_data(data(kk1,1:76), strcat(path,path_data_save(2),int2str(kk1-1),'.png'), true, [1.0, 2.0]);
    end
end