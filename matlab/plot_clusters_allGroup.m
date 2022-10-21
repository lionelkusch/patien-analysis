close('all');
clear('all');

%% load data
path = '/home/kusch/Documents/project/patient_analyse/';
load(strcat(path,'/projection_data/first_projection/all_subject_Y_phate_knn_dist_cosine_mds_dist_cosine_nb_comp_3_nb_pca_5_gamma_-1.0.mat'));


%% plot figures
% parameters 
groups=cluster_index+1; % index of clusters
d2 = false; % plot in 2d or 3 dimensions
plot = false; % save or not the figure

for kk1=1:max(groups)
   patterns(:,kk1)=sum(avalanches_binarize(groups==kk1,:),1);
   positions{kk1}=PHATE_position(groups==kk1,:);
end

if d2
    for kk1= 1:size(patterns,2)
        plot_data(patterns(1:78,kk1), strcat('../paper/result/default/plot_brain/cluster_',int2str(kk1-1),'.png'), plot)
        fig = figure;
        scatter(PHATE_position(:,1),PHATE_position(:,2));
        hold on
        scatter(positions{kk1}(:,1),positions{kk1}(:,2),'r');
        hold off
        fig.Position = [0 0 2000 2000];
        set(gcf, 'InvertHardCopy', 'off'); 
        set(gcf,'Color',[0 0 0]); % RGB values [0 0 0] indicates black color
        if (plot)
            saveas(gcf, strcat('../paper/result/default/plot_brain/cluster_2D_',int2str(kk1-1),'.png'), plot);
            close('all');
        end
    end    
else
     for kk1= 1:size(patterns,2)
        plot_data(patterns(1:78,kk1), strcat('../paper/result/default/plot_brain/cluster_',int2str(kk1-1),'.png'), plot)
        fig = figure;
        scatter3(PHATE_position(:,1),PHATE_position(:,2),PHATE_position(:,3));
        hold on
        scatter3(positions{kk1}(:,1),positions{kk1}(:,2),positions{kk1}(:,3),'r');
        hold off
        set(gcf, 'InvertHardCopy', 'off'); 
        set(gcf,'Color',[0 0 0]); % RGB values [0 0 0] indicates black color
        fig.Position = [0 0 2000 2000];
        if (plot)
            saveas(gcf, strcat('../paper/result/default/plot_brain/cluster_2D_',int2str(kk1-1),'.png'))
            close('all');
        end
     end
end

