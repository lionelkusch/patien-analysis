% load('C:\Users\ppsor\cluster_18_patients.mat')
% load('/home/kusch/Documents/project/patient_analyse/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_2_nb_pca_5_gamma_-1.0.mat')
% load('/home/kusch/Documents/project/patient_analyse/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_2_nb_pca_8_gamma_-1.0.mat')
load('/home/kusch/Documents/project/patient_analyse/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_3_nb_pca_5_gamma_-1.0.mat')
% load('/home/kusch/Documents/project/patient_analyse/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_3_nb_pca_6_gamma_-1.0.mat')
groups=cluster_index+1;
d2 = false;

for kk1=1:max(groups)
   patterns(:,kk1)=sum(avalanches_binarize(groups==kk1,:),1);
   positions{kk1}=PHATE_position(groups==kk1,:);
end

if d2
    for kk1= 1:size(patterns,2)
        plot_data(patterns(1:78,kk1))
        figure
        scatter(PHATE_position(:,1),PHATE_position(:,2));
        hold on
        scatter(positions{kk1}(:,1),positions{kk1}(:,2),'r');
        hold off
    end    
else
     for kk1= 1:size(patterns,2)
        plot_data(patterns(1:78,kk1))
        figure
        scatter3(PHATE_position(:,1),PHATE_position(:,2),PHATE_position(:,3));
        hold on
        scatter3(positions{kk1}(:,1),positions{kk1}(:,2),positions{kk1}(:,3),'r');
        hold off
     end
end


%%
function plot_data(data)
%addpath('C:\Users\ppsor\Desktop\Code\spm8');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/spm8')
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

