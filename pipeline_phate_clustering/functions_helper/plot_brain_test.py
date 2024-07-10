import pyvista as pv
import trimesh
import numpy as np
import matplotlib.pyplot as plt


def multiview_pyvista(axs, vertices, lhfaces, rhfaces, region_map, data, region_select=None, cmap='viridis'):
    if region_select is not None:
        if data.shape[0] == region_select.shape[0]:
            data_tmp = np.ones(vertices.shape[0]) * np.NAN
            for index, region_id in enumerate(region_select):
                data_tmp[np.where(region_map == region_id)] = data[index]
            data = data_tmp
        else:
            region_unique = np.unique(region_map)
            for region_id in region_unique:
                if region_id not in region_select:
                    data[np.where(region_map == region_id)] = np.NaN

    mesh = pv.wrap(trimesh.Trimesh(vertices, faces=np.concatenate((lhfaces, rhfaces)), process=False))
    mesh['color'] = data
    mesh_left = pv.wrap(trimesh.Trimesh(vertices, faces=lhfaces, process=False))
    mesh_left['color'] = data
    mesh_right = pv.wrap(trimesh.Trimesh(vertices, faces=rhfaces, process=False))
    mesh_right['color'] = data

    p = pv.Plotter(window_size=(400, 500), off_screen=True)
    p.set_background(color="white")
    p.add_mesh(mesh, cmap=cmap)
    p.view_xy()
    p.set_position([p.camera_position[0][0], p.camera_position[0][1], p.camera_position[0][2]-150])
    def my_cpos_callback_func(p1):
        def my_cpos_callback(*args):
            p1.add_text(str(p1.camera_position)+' '+str(p1.window_size), name="cpos", color='black')
            return
        return my_cpos_callback
    p.add_key_event("p", my_cpos_callback_func(p))
    screen_top = p.screenshot(return_img=True, transparent_background=False)[70:425, 60:340]

    p2 = pv.Plotter(window_size=(400, 500), off_screen=True)
    p2.set_background(color="white")
    p2.add_mesh(mesh_left, cmap=cmap)
    p2.view_yz()
    p2.set_position([p2.camera_position[0][0]-100, p2.camera_position[0][1], p2.camera_position[0][2]])
    # p2.add_key_event("p", my_cpos_callback_func(p2))
    # p2.show()
    screen_left_top = p2.screenshot(return_img=True, transparent_background=False)[105:390, 0:400]

    p3 = pv.Plotter(window_size=(400, 500), off_screen=True)
    p3.set_background(color="white")
    p3.add_mesh(mesh_left, cmap=cmap)
    p3.view_yz(negative=True)
    p3.set_position([p3.camera_position[0][0], p3.camera_position[0][1], p3.camera_position[0][2]])
    # p3.add_key_event("p", my_cpos_callback_func(p3))
    # p3.show()
    screen_left_bottom = p3.screenshot(return_img=True, transparent_background=False)[140:360, 50:345]


    p4 = pv.Plotter(window_size=(400, 500), off_screen=True)
    p4.set_background(color="white")
    p4.add_mesh(mesh_right, cmap=cmap)
    p4.view_yz()
    p4.set_position([p4.camera_position[0][0]-100, p4.camera_position[0][1], p4.camera_position[0][2]])
    # p4.add_key_event("p", my_cpos_callback_func(p2))
    # p4.show()
    screen_right_top = p4.screenshot(return_img=True, transparent_background=False)[110:390, 20:380]

    p5 = pv.Plotter(window_size=(400, 500), off_screen=True)
    p5.set_background(color="white")
    p5.add_mesh(mesh_right, cmap=cmap)
    p5.view_yz(negative=True)
    p5.set_position([p5.camera_position[0][0], p5.camera_position[0][1], p5.camera_position[0][2]])
    # p5.add_key_event("p", my_cpos_callback_func(p5))
    # p5.show()
    screen_right_bottom = p5.screenshot(return_img=True, transparent_background=False)[130:370, 40:360]

    # fig = plt.figure(figsize=(2.0, 1.0))
    axs[0].matshow(screen_left_bottom)
    axs[0].axis('off')
    axs[1].matshow(screen_left_top)
    axs[1].axis('off')
    axs[2].matshow(screen_right_top)
    axs[2].axis('off')
    axs[3].matshow(screen_right_bottom)
    axs[3].axis('off')
    axs[4].matshow(screen_top)
    axs[4].axis('off')


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


if __name__ == '__main__':
    import os
    import scipy.io as io

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
    multiview_pyvista(axs, vertices, lhfaces, rhfaces, region_map_tmp, data)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)
    for name in region_select[[0, 2]]:
        one_region = np.zeros(vertices.shape[0])
        one_region[np.where(region_map_tmp == name.strip())] = 1
        fig = plt.figure(figsize=(15, 10))
        axs = [plt.subplot(231), plt.subplot(234), plt.subplot(233), plt.subplot(236), plt.subplot(132)]
        multiview_pyvista(axs, vertices, lhfaces, rhfaces, region_map_tmp, one_region, region_select=region_select)
        plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)

    one_region = np.zeros(region_select.shape[0])
    one_region[0] = 1
    one_region[1] = 1
    fig = plt.figure(figsize=(15, 10))
    axs = [plt.subplot(231), plt.subplot(234), plt.subplot(233), plt.subplot(236), plt.subplot(132)]
    multiview_pyvista(axs, vertices, lhfaces, rhfaces, region_map_tmp, one_region, region_select=region_select)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0., wspace=0., hspace=0.)

    plt.show()