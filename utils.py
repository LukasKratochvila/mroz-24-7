import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

floorplan_map = {
    0: [255, 255, 255],  # background white
    1: [0, 0, 0],  # walls black
    2: [230, 25, 75],  # glass_walls red
    3: [60, 180, 75],  # railings green
    4: [255, 225, 25],  # doors yellow
    5: [0, 130, 200],  # sliding_doors blue
    6: [245, 130, 48],  # windows orange
    7: [70, 240, 240],  # stairs_all cyan
}

floorplan_map_rgba = {
    0: [255, 255, 255, 0],  # background white
    1: [0, 0, 0, 255],  # walls black
    2: [230, 25, 75, 255],  # glass_walls red
    3: [60, 180, 75, 255],  # railings green
    4: [255, 225, 25, 255],  # doors yellow
    5: [0, 130, 200, 255],  # sliding_doors blue
    6: [245, 130, 48, 255],  # windows orange
    7: [70, 240, 240, 255],  # stairs_all cyan
}


def ind2rgb(ind_img, color_map=None):
    if color_map is None:
        color_map = floorplan_map
    else:
        color_map = eval(color_map)
    rgb_img = np.zeros((ind_img.shape[0], ind_img.shape[1], 3), dtype=np.uint8)  # Be aware default color is [0, 0, 0]

    for i, rgb in color_map.items():
        rgb_img[(ind_img == i)] = rgb

    return rgb_img


def ind2rgba(ind_img, color_map=None):
    if color_map is None:
        color_map = floorplan_map_rgba
    rgb_img = np.zeros((ind_img.shape[0], ind_img.shape[1], 4), dtype=np.uint8)

    for i, rgb in color_map.items():
        rgb_img[(ind_img == i)] = rgb

    return rgb_img


def rgb2ind(img, color_map=None):
    if color_map is None:
        color_map = floorplan_map
    ind = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i, rgb in color_map.items():
        ind[(img == rgb).all(2)] = i

    return ind


def plot_legend():
    classes = ['background', 'walls', 'glass walls', 'railings', 'doors', 'sliding doors', 'windows', 'stairs']

    fig, ax = plt.subplots(dpi=400)
    patches = []
    for i, c in enumerate(classes):
        patches.append(mpatches.Patch(color=np.array(floorplan_map.get(i)) / 255, label=c[0].upper() + c[1:]))
    ax.legend(handles=patches)
    ax.axis("off")
    # ax.legend(handles=patches, ncol=len(classes))
    plt.savefig('legend', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot_legend()