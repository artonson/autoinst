# Different function to color image with projected point clouds, visualizing point clouds etc.
import numpy as np
import cv2
import matplotlib.pyplot as plt


def unite_pcd_and_img(point_to_pixel_matches: dict, pcd_camframe, img, coloring='depth'):
    '''
    Function that takes a dict that maps point indices to pixel coordinates and returns 
    an image with projected point clouds    
    Args:
        point_to_pixel_matches: dict that maps point indices to pixel coordinates
        pcd_camframe:           point clouds in camera frame
        img:                    image to be colored
        coloring:               color scheme, 'depth'
    Returns:
        img_with_pc:            image with projected point clouds
    '''

    if coloring == 'depth':
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        max_depth = np.max(pcd_camframe[:, 2])

    ### Iterate over point_to_pixel_matches values and color image accordingly
    img_with_pc = img.copy()
    for index, pixel in point_to_pixel_matches.items():

        if coloring == 'depth':
            depth = pcd_camframe[index, 2]
            id = min(int(255), int(255 * depth / max_depth))
            color = cmap[id, :]
        else:
            color = (255,0,0)

        cv2.circle(img_with_pc, (pixel[0], pixel[1]), 2, color=tuple(color), thickness=1)
    
    return img_with_pc
