import os 
import numpy as np
import open3d as o3d

#fn = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/theta_0.5_alpha_1.0_t_0.03.pcd'

#fn = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/3duis_7.pcd'

fn = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/output_chunks/test_data/000125.pcd'
fn0 = '/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/output_chunks/test_data/000093.pcd'



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


pcd = o3d.io.read_point_cloud(fn)
pcd0 = o3d.io.read_point_cloud(fn0)
pcds = [pcd0,pcd]
out = merge_chunks_unite_instances(pcds)
o3d.visualization.draw_geometries([out])


