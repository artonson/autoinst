# Different function to color image with projected point clouds, visualizing point clouds etc.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import random
import copy

from point_cloud_utils import get_pcd
from point_to_pixels import pixel_to_point_from_point_to_pixel

def generate_random_colors(N):
    colors = []
    for _ in range(N):
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    return colors


def unite_pcd_and_img(point_to_pixel_matches: dict, img, label_map=None, is_instance = False, coloring='depth', radius=2):
    '''
    Function that takes a dict that maps point indices to pixel coordinates and returns 
    an image with projected point clouds    
    Args:
        point_to_pixel_matches: dict that maps point indices to pixel coordinates
        pcd_camframe:           point clouds in camera frame
        img:                    image to be colored
        label_map:              label map of image
        is_instance:            whether the label map contains instance labels or colors
        coloring:               color scheme, 'depth' or 'label_map'
    Returns:
        img_with_pc:            image with projected point clouds
    '''

    if coloring == 'depth':
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        max_depth = 0
        for _, point_data in point_to_pixel_matches.items():
            depth = point_data['depth']
            if depth > max_depth:
                max_depth = depth

    if coloring == 'label_map':
        assert(label_map is not None)
        if is_instance:
            colors = generate_random_colors(len(np.unique(label_map))+1)

    ### Iterate over point_to_pixel_matches values and color image accordingly
    img_with_pc = img.copy()

    ### Initalize a dict, in which the closest point to each pixel is stored
    pixel_to_point_matches = pixel_to_point_from_point_to_pixel(point_to_pixel_matches)

    for pixel, pixel_data in pixel_to_point_matches.items():

        if coloring == 'depth':
            depth = pixel_data['depth']
            id = min(int(255), int(255 * depth / max_depth))
            color = cmap[id, :]
        elif coloring == 'label_map':
            if is_instance:
                instance_label = int(label_map[pixel[1], pixel[0]])
                if instance_label: # ignore unlabeled pixels
                    color = colors[instance_label]
                else: 
                    continue
            else:
                color = label_map[pixel[1], pixel[0]].tolist()
                if color == [70,70,70]: # ignore unlabeled pixels
                    continue
        else:
            color = (255,0,0)

        cv2.circle(img_with_pc, (pixel[0], pixel[1]), radius, color=tuple(color), thickness=-1)
    
    return img_with_pc


def color_pcd_with_labels(points: np.array, point_to_label: dict):
    '''
    Args:
        points:                  3D points in camera coordinate [npoints, 3]
        point_to_label:          dict that maps point indices to instance label color
    Returns:
        colored_pcd:             colored point cloud
    '''
    colored_pcd = get_pcd(points)

    colors = []
    for color in point_to_label.values():
        colors.append(list(color))

    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    return colored_pcd

def visualize_associations_in_img(_label, associations):
    '''
    Args:
        _label:             label map of image
        associations:       dict that maps original label colors to associated label colors
    Returns:
        label:              label map of image with instance labels
    '''

    label = _label.copy()
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color_norm = tuple(label[i,j]/255.0)

            association = associations.get(color_norm)

            if association is not None:
                color = list(association)
                label[i,j] = [i * 255.0 for i in color]

            else:
                label[i,j] = [0,0,0]
    return label


def color_pcd_by_labels(pcd, labels):
    
    colors = generate_random_colors(500)
    pcd_colored = copy.deepcopy(pcd)
    pcd_colored.colors = o3d.utility.Vector3dVector(np.zeros(np.asarray(pcd.points).shape))

    for i in range(len(pcd_colored.points)):
        if labels[i] != (-1):
            pcd_colored.colors[i] = np.array(colors[labels[i]]) / 255

    return pcd_colored