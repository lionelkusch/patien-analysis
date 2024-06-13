from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as io


def multiview_matplot(axs, vtx, lh_tri, rh_tri, region_map, data, suptitle='', region_select=None, **kwds):
    x, y, z = vtx.T
    lh_x, lh_y, lh_z = vtx.T
    lh_tx, lh_ty, lh_tz = vtx[lh_tri].mean(axis=1).T
    rh_x, rh_y, rh_z = vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T

    if region_select is not None:
        if data.shape[0] == region_select.shape[0]:
            data_tmp = np.ones(vtx.shape[0]) * np.NAN
            for index, region_id in enumerate(region_select):
                data_tmp[np.where(region_map == region_id)] = data[index]
            data = data_tmp
        else:
            region_unique = np.unique(region_map)
            for region_id in region_unique:
                if region_id not in region_select:
                    data[np.where(region_map == region_id)] = np.NaN



    views = {
        'lh-lateral': Triangulation(-lh_y, lh_z, lh_tri[np.argsort(lh_tx)[::-1]]),
        'lh-medial': Triangulation(lh_y, lh_z, lh_tri[np.argsort(lh_tx)]),
        'rh-medial': Triangulation(-rh_y, rh_z, rh_tri[np.argsort(rh_tx)[::-1]]),
        'rh-lateral': Triangulation(rh_y, rh_z, rh_tri[np.argsort(rh_tx)]),
        'both-superior': Triangulation(x, y, np.concatenate((lh_tri, rh_tri))[np.argsort(np.concatenate((lh_tz,rh_tz)))])
    }

    def plotview(ax, viewkey, z=None, zmin=None, zmax=None, zthresh=None, suptitle='', shaded=True,
                 cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        if z is None:
            z = np.random.rand(v.x.shape[0])
        if not viewlabel:
            ax.axis('off')
        if zmin is None:
            zmin = np.nanmin(z)
        if zmax is None:
            zmax = np.nanmax(z)
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        tc.set_clim(vmin=zmin, vmax=zmax)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            plt.xlabel(viewkey)
        return tc

    plotview(axs[0], 'lh-lateral', data, **kwds)
    plotview(axs[1], 'lh-medial', data, **kwds)
    plotview(axs[2], 'rh-lateral', data, **kwds)
    plotview(axs[3], 'rh-medial', data, **kwds)
    tc = plotview(axs[4], 'both-superior', data, suptitle=suptitle, **kwds)
    return tc


def get_region_select():
    region_select = np.array([
        'Rectus_L            ',
        'Olfactory_L         ',
        'Frontal_Sup_Orb_L   ',
        'Frontal_Med_Orb_L   ',
        'Frontal_Mid_Orb_L   ',
        'Frontal_Inf_Orb_L   ',
        'Frontal_Sup_L       ',
        'Frontal_Mid_L       ',
        'Frontal_Inf_Oper_L  ',
        'Frontal_Inf_Tri_L   ',
        'Frontal_Sup_Medial_L',
        'Supp_Motor_Area_L   ',
        'Paracentral_Lobule_L',
        'Precentral_L        ',
        'Rolandic_Oper_L     ',
        'Postcentral_L       ',
        'Parietal_Sup_L      ',
        'Parietal_Inf_L      ',
        'SupraMarginal_L     ',
        'Angular_L           ',
        'Precuneus_L         ',
        'Occipital_Sup_L     ',
        'Occipital_Mid_L     ',
        'Occipital_Inf_L     ',
        'Calcarine_L         ',
        'Cuneus_L            ',
        'Lingual_L           ',
        'Fusiform_L          ',
        'Heschl_L            ',
        'Temporal_Sup_L      ',
        'Temporal_Mid_L      ',
        'Temporal_Inf_L      ',
        'Temporal_Pole_Sup_L ',
        'Temporal_Pole_Mid_L ',
        'ParaHippocampal_L   ',
        'Cingulum_Ant_L      ',
        'Cingulum_Mid_L      ',
        'Cingulum_Post_L     ',
        'Insula_L            ',
        'Rectus_R            ',
        'Olfactory_R         ',
        'Frontal_Sup_Orb_R   ',
        'Frontal_Med_Orb_R   ',
        'Frontal_Mid_Orb_R   ',
        'Frontal_Inf_Orb_R   ',
        'Frontal_Sup_R       ',
        'Frontal_Mid_R       ',
        'Frontal_Inf_Oper_R  ',
        'Frontal_Inf_Tri_R   ',
        'Frontal_Sup_Medial_R',
        'Supp_Motor_Area_R   ',
        'Paracentral_Lobule_R',
        'Precentral_R        ',
        'Rolandic_Oper_R     ',
        'Postcentral_R       ',
        'Parietal_Sup_R      ',
        'Parietal_Inf_R      ',
        'SupraMarginal_R     ',
        'Angular_R           ',
        'Precuneus_R         ',
        'Occipital_Sup_R     ',
        'Occipital_Mid_R     ',
        'Occipital_Inf_R     ',
        'Calcarine_R         ',
        'Cuneus_R            ',
        'Lingual_R           ',
        'Fusiform_R          ',
        'Heschl_R            ',
        'Temporal_Sup_R      ',
        'Temporal_Mid_R      ',
        'Temporal_Inf_R      ',
        'Temporal_Pole_Sup_R ',
        'Temporal_Pole_Mid_R ',
        'ParaHippocampal_R   ',
        'Cingulum_Ant_R      ',
        'Cingulum_Mid_R      ',
        'Cingulum_Post_R     ',
        'Insula_R            ',
    ])
    for index in range(len(region_select)):
        region_select[index] = region_select[index].strip()
    return region_select

def get_brain_mesh():
    path = os.path.dirname(os.path.realpath(__file__)) + '/../../matlab/library/'
    lhfaces = np.array(io.loadmat(path +'/faces.mat')['lhfaces'] - 1, dtype=int)
    rhfaces = np.array(io.loadmat(path +'/faces.mat')['rhfaces'] - 1, dtype=int)
    vertices = io.loadmat(path +'/vertices.mat')['vertices']
    region_map_tmp = io.loadmat(path +'/region_map.mat')['region_map']
    region_map_tmp = np.ascontiguousarray(np.array(region_map_tmp, dtype=int).view('U1')[0::2])
    region_map_tmp = np.array([''.join(row).strip() for row in region_map_tmp])
    return lhfaces, rhfaces, vertices, region_map_tmp



if __name__ == '__main__':

    region_select = get_region_select()

    ROI_indices = [27, 21, 5, 25, 9, 15, 3, 7, 11, 13, 23, 19, 69, 1, 17, 57, 59, 61, 63, 65, 67, 49, 51, 53, 43, 45,
                   47, 55, 79, 81, 85, 89, 83, 87, 39, 31, 33, 35, 29,
                   28, 22, 6, 26, 10, 16, 4, 8, 12, 14, 24, 20, 70, 2, 18, 58, 60, 62, 64, 66, 68, 50, 52, 54, 44, 46,
                   48, 56, 80, 82, 86, 90, 84, 88, 40, 32, 34, 36, 30]


    path = os.path.dirname(os.path.realpath(__file__)) + '/../../matlab/library/'
    ROI = io.loadmat(path + 'ROI_MNI_V4_List.mat')
    all_region = np.concatenate(ROI['ROI']['Nom_L'][0])
    lhfaces = np.array(io.loadmat(path +'/faces.mat')['lhfaces'] - 1, dtype=int)
    rhfaces = np.array(io.loadmat(path +'/faces.mat')['rhfaces'] - 1, dtype=int)
    vertices = io.loadmat(path +'/vertices.mat')['vertices']
    region_map_tmp = io.loadmat(path +'/region_map.mat')['region_map']
    region_map_tmp = np.ascontiguousarray(np.array(region_map_tmp, dtype=int).view('U1')[0::2])
    region_map_tmp = np.array([''.join(row).strip() for row in region_map_tmp])
    hemishpere = []
    for index, row in enumerate(region_map_tmp):
        if len(row) != 0:
            hemishpere.append('L' == row[-1])
        else:
            hemishpere.append(-1)
    hemishpere = np.array(hemishpere, dtype=int)

    cmap = plt.cm.coolwarm
    cmap.set_bad(color='grey')

    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(231), plt.subplot(234), plt.subplot(233), plt.subplot(236), plt.subplot(132)]
    data = np.random.rand(vertices.shape[0])
    multiview_matplot(axs, vertices, lhfaces, rhfaces, region_map_tmp, data, cmap=cmap, shaded=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)

    for name in region_select[[0, 2]]:
        one_region = np.zeros(vertices.shape[0])
        one_region[np.where(region_map_tmp == name.strip())] = 1
        fig = plt.figure(figsize=(15, 10))
        axs = [plt.subplot(231), plt.subplot(234), plt.subplot(233), plt.subplot(236), plt.subplot(132)]
        multiview_matplot(axs, vertices, lhfaces, rhfaces, region_map_tmp, one_region, suptitle=name.strip(),
                  region_select=region_select, cmap=cmap, shaded=False)
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)

    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(231), plt.subplot(234), plt.subplot(233), plt.subplot(236), plt.subplot(132)]
    one_region = np.zeros(region_select.shape[0])
    one_region[0] = 1
    one_region[1] = 1
    multiview_matplot(axs, vertices, lhfaces, rhfaces, region_map_tmp, one_region,
              region_select=region_select, cmap=cmap, shaded=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)


    plt.show()