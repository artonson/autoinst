# Different function to color image with projected point clouds, visualizing point clouds etc.
import numpy as np
import cv2
import matplotlib.pyplot as plt


def unite_pcd_and_img(point_to_pixel_matches: dict, img, coloring='depth'):
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

    max_depth = 0

    ### Iterate over point_to_pixel_matches values and color image accordingly
    img_with_pc = img.copy()

    ### Initalize a dict, in which the closest point to each pixel is stored
    pixel_to_point_matches = {}

    for index, point_data in point_to_pixel_matches.items():
        pixel = point_data['pixels']
        pixel_tpl = (pixel[0], pixel[1])
        depth = point_data['depth']

        if depth > max_depth:
            max_depth = depth

        if pixel_tpl not in pixel_to_point_matches.keys():
            # Add the pixel and point index to the dictionary
            pixel_to_point_matches[pixel_tpl] = {'index': index, 'depth': depth}
        elif depth < pixel_to_point_matches[pixel_tpl]['depth']:
            # Update the dictionary with the new point index and depth
            pixel_to_point_matches[pixel_tpl] = {'index': index, 'depth': depth}
        

    for pixel, pixel_data in pixel_to_point_matches.items():
        index = pixel_data['index']

        if coloring == 'depth':
            depth = pixel_data['depth']
            id = min(int(255), int(255 * depth / max_depth))
            color = cmap[id, :]
        else:
            color = (255,0,0)

        cv2.circle(img_with_pc, (pixel[0], pixel[1]), 2, color=tuple(color), thickness=1)
    
    return img_with_pc
