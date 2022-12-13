close('all');
clear('all');
% addpath('C:\Users\ppsor\Desktop\Code\spm8');
% addpath('C:\Users\ppsor\Desktop\Code\brain_image');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/');
addpath('/home/kusch/Documents/project/patient_analyse/matlab/library/spm8/');
%% load data
% path = 'C:\Users\ppsor\Desktop\studi\Lionel\brains\';
path = '/home/kusch/Documents/project/patient_analyse//paper/result/default/';
path_save = strcat(path, '/figure/');
load(strcat(path,'data_with_cluster.mat'));
d2 = false; % plot in 2d or 3 dimensions
plot = true; % save or not the figure

%concatenate avalanches
avalanches_binarize=vertcat(avalanches_binarize{:});

for kk_clust=1:size(cluster,1)
    %% plot figures
    % parameters
    groups=cluster{kk_clust,2}+1; % index of clusters

    for kk1=1:max(groups)
       patterns(:,kk1)=sum(avalanches_binarize(groups==kk1,:),1);
       positions{kk1}=PHATE_position(groups==kk1,:);
    end

    if d2
        % for the 2 dimensionals cluster
        for kk1= 1:size(patterns,2)
             % plot brain cluster
             plot_data(patterns(1:78,kk1), strcat(path_save, '\k_',int2str(cluster{kk_clust,1}),'cluster_',int2str(kk1-1),'.png'), plot)
             % plot cluster in low dimension/Phate dimension
             if (plot)
                 fig = figure('visible','off');
             else
                fig = figure;
             end
             scatter(PHATE_position(:,1),PHATE_position(:,2));
             hold on
             scatter(positions{kk1}(:,1),positions{kk1}(:,2),'r');
             hold off
             fig.Position = [0 0 2000 2000];
             set(gcf, 'InvertHardCopy', 'off');
             set(gcf,'Color',[0 0 0]); % RGB values [0 0 0] indicates black color
             if (plot)
                 saveas(gcf, strcat(path_save, '\k_',int2str(cluster{kk_clust,1}),'cluster2D_',int2str(kk1-1),'.png'), plot);
                 close('all');
             end
        end
    else
        % for 3 dimensional cluster
         for kk1= 1:size(patterns,2)
             % plot brain cluster
             plot_data(patterns(1:78,kk1),strcat(path_save, '\k_',int2str(cluster{kk_clust,1}),'cluster_',int2str(kk1-1),'.png'), plot)
             % plot cluster in low dimension/Phate dimension
             if (plot)
                 fig = figure('visible','off');
             else
                fig = figure;
             end
             scatter3(PHATE_position(:,1),PHATE_position(:,2),PHATE_position(:,3));
             hold on
             scatter3(positions{kk1}(:,1),positions{kk1}(:,2),positions{kk1}(:,3),'r');
             hold off
             set(gcf, 'InvertHardCopy', 'off');
             set(gcf,'Color',[0 0 0]); % RGB values [0 0 0] indicates black color
             fig.Position = [0 0 2000 2000];
             if (plot)
                 saveas(gcf, strcat(path_save, '\k_',int2str(cluster{kk_clust,1}),'cluster2D_',int2str(kk1-1),'.png'))
                 close('all');
             end
         end
    end
end