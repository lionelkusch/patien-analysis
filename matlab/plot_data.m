function plot_data(data, file_name, plot, colour_range)

colourbar_threshold=[]; % can be used to adjust the colour range (experimental)
mesh_type = 'spm_canonical'; % assume that input contains 78 AAL ROIs
nr_views=6; % #views of the cortical surface in the figures
mesh_labels = 'AAL';
if ~exist('colour_range','var'), colour_range=[]; end % for display: colour_range will be based on the data; alternatively, you can provide a maximum and minimum value
if ~exist('plot','var'), plot=false; end

display(colour_range);

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
    
    if (plot)
        fig = gcf;
        fig.Position = [0 0 2000 2000];
        saveas(gcf, file_name)
        close('all')
    end
end