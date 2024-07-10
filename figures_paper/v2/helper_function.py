import numpy as np
import matplotlib.patches as mpatches


def add_arrows(data_x, data_y, ax, xlabel, ylabel):
    min_x = np.min(data_x)
    max_x = np.max(data_x)
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = np.min(data_y)
    max_y = np.max(data_y)
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    arr_1 = mpatches.FancyArrowPatch((min_x, min_y), (0.35 * (max_x - min_x) + min_x, min_y),
                                     arrowstyle='->,head_width=.15', mutation_scale=20)
    arr_1.set_clip_on(False)
    ax.add_patch(arr_1)
    ax.annotate(xlabel, (0., -0.1), xycoords=arr_1, ha='left', va='top', annotation_clip=False)
    arr_2 = mpatches.FancyArrowPatch((min_x, min_y), (min_x, 0.35 * (max_y - min_y) + min_y),
                                     arrowstyle='->,head_width=.15', mutation_scale=20)
    arr_2.set_clip_on(False)
    ax.add_patch(arr_2)
    ax.annotate(ylabel, (0.0, 0.5), xycoords=arr_2, ha='right', va='center', annotation_clip=False, rotation=90)
    ax.set_ylim(ymin=min_y, ymax=max_y)
    ax.set_xlim(xmin=min_x, xmax=max_x)
