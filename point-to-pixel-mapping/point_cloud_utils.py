# Transformation on point clouds
import numpy as np
import open3d as o3d
import copy
from open3d.pipelines import registration

def get_pcd(points: np.array):
    """
    Convert numpy array to open3d point cloud
    Args:
        points: 3D points in camera coordinate [npoints, 3] or [npoints, 4]
    Returns:
        pcd: open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

def transform_pcd(points, T):
    """
    Apply the perspective projection
    Args:
        points:              3D points in coordinate system 1 [npoints, 3]
        T:                   Transformation matrix in homogeneous coordinates [4, 4]
    Returns:
        points:              3D points in coordinate system 2 [npoints, 3]
    """
    pcd = get_pcd(points)
    transformed_pcd = pcd.transform(T)
    return np.asarray(transformed_pcd.points)

def filter_points_from_dict(points, filter_dict):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_dict.keys()))
    return points[inds]

def filter_points_from_list(points, filter_list):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_list))
    return points[inds]

def point_to_label(point_to_pixel: dict, label_map, label_is_color=True, label_is_instance=False):
    '''
    Args:
        point_to_pixel: dict that maps point indices to pixel coordinates
        label_map: label map of image
    Returns:
        point_to_label: dict that maps point indices to labels
    '''
    point_to_label_dict = {}

    for index, point_data in point_to_pixel.items():
        pixel = point_data['pixels']

        if label_is_color:
            color = label_map[pixel[1], pixel[0]]
            if color.tolist() == [70,70,70]: # ignore unlabeled pixels
                continue
            point_to_label_dict[index] = tuple((color / 255))

        elif label_is_instance:
            instance_label = int(label_map[pixel[1], pixel[0]])
            if instance_label: # ignore unlabeled pixels
                point_to_label_dict[index] = instance_label
            else:
                continue
            
        else: # label is continuous feature vector
            point_to_label_dict[index] = label_map[pixel[1], pixel[0]]
    
    return point_to_label_dict

def change_point_indices(point_to_X: dict, indices: list):
    '''
    Args:
        point_to_X: dict that maps point indices to X
        indices: list of indices
    Returns:
        point_to_X: dict that maps point indices to X
    '''
    point_to_X_new = {}
    for old_index in point_to_X.keys():
        new_index = indices[old_index]
        point_to_X_new[new_index] = point_to_X[old_index]
    
    return point_to_X_new

def transformation_matrix(rotation: np.array, translation: np.array):
    """
    Create a transformation matrix from rotation and translation
    Args:
        rotation:    Rotation matrix [3, 3]
        translation: Translation vector [3, 1]
    Returns:
        T:           Transformation matrix in homogeneous coordinates [4, 4]
    """

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def kDTree_1NN_feature_reprojection(features_to, pcd_to, features_from, pcd_from, max_radius=None, no_feature_label=[1,0,0]):
    '''
    Args:
        pcd_from: point cloud to be projected
        pcd_to: point cloud to be projected to
        search_method: search method ("radius", "knn")
        search_param: search parameter (radius or k)
    Returns:
        features_to: features projected on pcd_to
    '''
    from_tree = o3d.geometry.KDTreeFlann(pcd_from)

    for i, point in enumerate(np.asarray(pcd_to.points)):

        [_, idx, _] = from_tree.search_knn_vector_3d(point, 1)
        if max_radius is not None:
            if np.linalg.norm(point - np.asarray(pcd_from.points)[idx[0]]) > max_radius:
                features_to[i,:] = no_feature_label
            else:
                features_to[i,:] = features_from[idx[0]]
        else:
            features_to[i,:] = features_from[idx[0]]
    
    return features_to


def remove_isolated_points(pcd, adjacency_matrix):

    isolated_mask = ~np.all(adjacency_matrix == 0, axis=1)
    adjacency_matrix = adjacency_matrix[isolated_mask][:, isolated_mask]
    pcd = pcd.select_by_index(np.where(isolated_mask == True)[0])

    return pcd, adjacency_matrix


def get_statistical_inlier_indices(pcd, nb_neighbors=20, std_ratio=2.0):
    _, inlier_indices = copy.deepcopy(pcd).remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return inlier_indices

def get_subpcd(pcd, indices, colors=False, normals=False):
    subpcd = o3d.geometry.PointCloud()
    subpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    if colors:
        subpcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    if normals:
        subpcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices])
    return subpcd


def merge_chunks_unite_instances(chunks: list, icp=False):
    last_chunk = chunks[0] 
    merge = o3d.geometry.PointCloud()
    merge += last_chunk

    for new_chunk in chunks[1:]:

        if icp:
            last_chunk.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
            new_chunk.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
            reg_p2l = registration.registration_icp(new_chunk, last_chunk, 0.9, np.eye(4), registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))
            transform = reg_p2l.transformation
            new_chunk.transform(transform)

        points_1 = np.asarray(last_chunk.points)
        points_2 = np.asarray(new_chunk.points)

        colors_1 = np.asarray(last_chunk.colors)
        colors_2 = np.asarray(new_chunk.colors)

        unique_colors_1 = np.unique(colors_1, axis=0)
        unique_colors_2 = np.unique(colors_2, axis=0)

        instance2point_1 = {}
        for i in range(unique_colors_1.shape[0]):
            if not np.all(unique_colors_1[i] == 0.0): # Streets are black
                instance2point_1[i] = {}
                inds = np.where(np.all(colors_1 == unique_colors_1[i], axis=1))[0]
                instance2point_1[i]["points"] = points_1[inds]
                instance2point_1[i]["inds"] = inds

        instance2point_2 = {}
        for i in range(unique_colors_2.shape[0]):
            if not np.all(unique_colors_2[i] == 0.0): # Streets are black
                instance2point_2[i] = {}
                inds = np.where(np.all(colors_2 == unique_colors_2[i], axis=1))[0]
                instance2point_2[i]["points"] = points_2[inds]
                instance2point_2[i]["inds"] = inds
        
        id_pairs_iou = []
        for id_1, entries_1 in instance2point_1.items():
            points1 = entries_1["points"]
            min_bound = np.min(points1, axis=0)
            max_bound = np.max(points1, axis=0)
            association = []
            for id_2, entries_2 in instance2point_2.items():
                points2 = entries_2["points"]
                intersection = np.where(np.all(points2 >= min_bound, axis=1) & np.all(points2 <= max_bound, axis=1))[0].shape[0]
                if intersection > 0:
                    union = len(np.unique(np.concatenate((points1, points2))))
                    iou = float(intersection) / float(union)
                    if iou > 0.01:
                        association.append((id_2, iou))
            if len(association) != 0:
                for (association_id, iou) in association:
                    id_pairs_iou.append((id_1, (association_id, iou)))
        
        ids_chunk_1 = []
        ids_chunk_2 = []
        ious = []
        for id1, (id2, iou) in id_pairs_iou:
            if id2 not in ids_chunk_2:
                ids_chunk_1.append(id1)
                ids_chunk_2.append(id2)
                ious.append(iou)
            else:
                i = ids_chunk_2.index(id2)
                if iou > ious[i]:
                    ious[i] = iou
                    ids_chunk_1[i] = id1

        for id1, id2 in zip(ids_chunk_1, ids_chunk_2):
            inds2 = instance2point_2[id2]["inds"]
            colors_2[inds2] = unique_colors_1[id1]

        new_chunk_recolored = o3d.geometry.PointCloud()
        new_chunk_recolored.points = new_chunk.points
        new_chunk_recolored.colors = o3d.utility.Vector3dVector(colors_2)
        last_chunk = new_chunk_recolored

        merge += new_chunk_recolored
        merge.remove_duplicated_points()

    return merge

def merge_unite_gt(chunks):
    last_chunk = chunks[0] 
    merge = o3d.geometry.PointCloud()
    merge += last_chunk

    for new_chunk in chunks[1:]:
        merge += new_chunk
    
    merge.remove_duplicated_points()
    return merge 
    
    


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / (np.linalg.norm(vector) + 1e-6)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))