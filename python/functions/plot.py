import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.cluster import KMeans

cnames = {
    # 'aliceblue': '#F0F8FF',
    # 'antiquewhite': '#FAEBD7',
    # 'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    # 'azure': '#F0FFFF',
    # 'beige': '#F5F5DC',
    # 'bisque': '#FFE4C4',
    'black': '#000000',
    # 'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    # 'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    # 'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    # 'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    # 'gainsboro': '#DCDCDC',
    # 'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    # 'greenyellow': '#ADFF2F',
    # 'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    # 'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'}


def plot_figure_3D_Kmeans(data, title):
    cluster_point = KMeans(n_clusters=5, random_state=123).fit_predict(data)
    fig = plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis('off')
    ax_1 = fig.add_subplot(121, projection='3d')
    ax_1.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.arange(data.shape[0]), s=0.8)
    ax_1.set_title('time')
    ax_2 = fig.add_subplot(122, projection='3d')
    ax_2.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_point, s=0.8)
    ax_2.set_title('cluster')
    plt.show()


def plot_figure_2D_Kmeans(data, title):
    cluster_point = KMeans(n_clusters=5, random_state=123).fit_predict(data)
    fig = plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis('off')
    ax_1 = fig.add_subplot(121)
    ax_1.scatter(data[:, 0], data[:, 1], c=np.arange(data.shape[0]), s=0.8)
    ax_1.set_title('time')
    ax_2 = fig.add_subplot(122)
    ax_2.scatter(data[:, 0], data[:, 1], c=cluster_point, s=0.8)
    ax_2.set_title('cluster')


def plot_figure_2D_patient(data, title, avalanches):
    fig = plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis('off')
    ax_1 = fig.add_subplot(121)
    ax_1.set_title('time')
    ax_2 = fig.add_subplot(122)
    ax_2.set_title('cluster')
    begin = 0
    for index, avalanche in enumerate(avalanches):
        end = begin + len(avalanche)
        ax_1.scatter(data[begin:end, 0], data[begin:end, 1], c=np.arange(len(avalanche)), s=0.8)
        ax_2.scatter(data[begin:end, 0], data[begin:end, 1], c=list(cnames.values())[index], s=0.8)
        begin = end


def plot_figure_2D(data, title, cluster_point):
    fig = plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis('off')
    ax_1 = fig.add_subplot(121)
    ax_1.scatter(data[:, 0], data[:, 1], c=np.arange(data.shape[0]), s=0.8)
    ax_1.set_title('time')
    ax_2 = fig.add_subplot(122)
    plot = ax_2.scatter(data[:, 0], data[:, 1], c=cluster_point, s=0.8, cmap='Pastel1')
    fig.colorbar(plot)
    ax_2.set_title('cluster')


def plot_figure_3D(data, title, cluster_point):
    fig = plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.axis('off')
    ax_1 = fig.add_subplot(121, projection='3d')
    ax_1.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.arange(data.shape[0]), s=0.8)
    ax_1.set_title('time')
    ax_2 = fig.add_subplot(122, projection='3d')
    a = ax_2.scatter(data[:, 0], data[:, 1], data[:, 2], c=cluster_point, s=0.8, cmap='tab20')
    ax_2.set_title('cluster')
    plt.colorbar(a)
    plt.show()


def plot_figure_2D_patient_unique(data, title, avalanches, figsize=(20, 20)):
    nb_grid = int(np.ceil(np.sqrt(len(avalanches))))
    fig, axs = plt.subplots(nb_grid, nb_grid, figsize=(20, 20))
    plt.title(title)
    plt.axis('off')
    begin = 0
    for index, avalanche in enumerate(avalanches):
        end = begin + len(avalanche)
        axs[0, 0].scatter(data[begin:end, 0], data[begin:end, 1], c=list(cnames.values())[index], s=0.8)
        axs[(index+1) % nb_grid, (index+1) // nb_grid].scatter(data[begin:end, 0], data[begin:end, 1],
                                                      c=list(cnames.values())[index], s=0.8)
        begin = end

def plot_figure_2D_patient_unique_time(data, title, avalanches, figsize=(20, 20)):
    nb_grid = int(np.ceil(np.sqrt(len(avalanches))))
    fig, axs = plt.subplots(nb_grid, nb_grid, figsize=(20, 20))
    plt.title(title)
    plt.axis('off')
    begin = 0
    for index, avalanche in enumerate(avalanches):
        end = begin + len(avalanche)
        axs[0, 0].scatter(data[begin:end, 0], data[begin:end, 1], c=np.arange(len(avalanche)), s=0.8)
        axs[(index+1) % nb_grid, (index+1) // nb_grid].scatter(data[begin:end, 0], data[begin:end, 1],
                                                      c=np.arange(len(avalanche)), s=0.8)
        begin = end
