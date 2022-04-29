clear all
close all
% load('C:\Users\ppsor\Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_4_nb_pca_5_gamma_-1.0.mat')
% load('C:\Users\ppsor\Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_4_nb_pca_15_gamma_0.0.mat')
% load('C:\Users\ppsor\Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_4_nb_pca_15_gamma_1.0.mat')
% load('C:\Users\ppsor\Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_2_nb_pca_15_gamma_-1.0.mat')
% load('C:\Users\ppsor\Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_2_nb_pca_10_gamma_-1.0.mat')
% load('C:\Users\ppsor\subject_1.mat')

out=cell(1,nb_cluster);
for kk1=1:nb_cluster   
    out{kk1}=avalanches_binarize(kmeans==kk1-1,:)    ;
end    

out2=zeros(nb_cluster,78);
for kk1=1:nb_cluster    
% out2(kk1,:)=zscore(sum(out{kk1}(:,1:78),1),[],2); 
out2(kk1,:)=[zscore(sum(out{kk1}(:,1:39),1),[],2),zscore(sum(out{kk1}(:,40:78),1),[],2)];
end    

% for kk1=1:nb_cluster 
% plot_data(out2(kk1,:))
% 
% end
load('C:\Users\ppsor\Desktop\studi\TCM\avalanches\avalanches_allbins_sameleng_AAL');
avalanches=avalanches{3}{6};
out=zeros(90,18);
idx=1;
for kk1=[47,43,42,39,38,33,30,29,24,23,22,20,18,16,15,10,9,7,6]; out(:,idx)=sum([avalanches{kk1}{:}],2); idx=idx+1; end; figure; plot(sum(out,2))
plot_data(sum(out(1:78,:),2))
function plot_data(data)
addpath('C:\Users\ppsor\Desktop\Code\spm8');
colourbar_threshold=[]; % can be used to adjust the colour range (experimental)
mesh_type = 'spm_canonical'; % assume that input contains 78 AAL ROIs
nr_views=6; % #views of the cortical surface in the figures
colour_range=[]; % for display: colour_range will be based on the data; alternatively, you can provide a maximum and minimum value
mesh_labels = 'AAL';

%% get AAL labels
[aalID, aalind,fullnames,everyID,allnames] = aal_get_numbers( 'Precentral_L' );
        tmplabels = char(allnames);
        cfg.allnames=tmplabels;
        
% Use only the most superfial areas
indices_in_same_order_as_in_Brainwave = select_ROIs_from_full_AAL(cfg);
labels = tmplabels(indices_in_same_order_as_in_Brainwave,:); %78 labels
    %% plot
    [colourbar_handle, patch_handles] = PaintBrodmannAreas_new2_clean(labels, data, length(data),length(data),nr_views, colour_range, colourbar_threshold, mesh_type, mesh_labels);
    set(gcf,'Tag','ShowBrainFigure');
    
    %display_label = deblank(labels(i,:));
    %display_label = strrep(display_label, '_', '\_');
    %title(sprintf('ROI %d: %s',i, display_label))
    drawnow
%     figfname = sprintf('ROI%d_%s.jpg', i, deblank(labels(i,:)))
%     export_fig(figfname, '-jpg')
%     close(gcf)
%     drawnow
    %%
end