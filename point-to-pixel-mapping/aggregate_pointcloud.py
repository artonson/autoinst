import open3d as o3d
import numpy as np
from open3d.pipelines import registration
from point_cloud_utils import get_pcd, get_subpcd, get_statistical_inlier_indices
import sys
#lib_path = '/Users/cedric/Lidar_Segmentation_Clustering/voxel_clustering_dependencies/build/patchworkpp/'
lib_path = '/home/cedric/unsup_3d_instances/pipeline/segmentation/utils/voxel_clustering_dependencies/build/patchworkpp/'
sys.path.insert(0,lib_path)
import pypatchworkpp
from visualization_utils import * 
from tqdm import tqdm 

def aggregate_pointcloud(dataset, ind_start, ind_end, icp=False, icp_threshold=0.9, ground_segmentation=None):
    '''
    Returns aggregated point cloud from dataset in world coordinate system
    Args:
        dataset:                dataset object
        ind_start:              start index of point clouds to aggregate
        ind_end:                end index of point clouds to aggregate
        icp:                    whether to use icp for registration
        icp_threshold:          threshold for icp
        ground_segmentation:    None, 'patchwork' or 'open3d'
    Returns:    
        map_pcd_ground:         Merged point cloud of ground points
        map_pcd_nonground:      Merged point cloud of non-ground points
        poses:                  List of poses of point clouds
        world_pose:             Pose of world coordinate system, i.e. of aggregated point cloud

    '''
    poses = []
    world_pose = np.eye(4)
    
    #pantopic_labels = []
    panoptic_labels_nonground = []
    panoptic_labels_ground = []
    seg_labels_nonground = []
    seg_labels_ground = []
    instance_labels_nonground = []
    instance_labels_ground = []
    moving_ids = [] ##for nuscenes 
    pose_id_dict = {} #for nuscenes to do moving object filtering 
    pcd_grounds = []
    pcd_non_grounds = []

    if ground_segmentation is None:

        map_pcd = o3d.geometry.PointCloud()

        for i in range(ind_start, ind_end):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dataset[i].point_cloud[:, :3])
            
            pose = dataset.get_pose(i)
            poses.append(pose)

            transform = pose

            if icp and i != ind_start:

                map_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))

                reg_p2l = registration.registration_icp(pcd, map_pcd, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))

                transform = reg_p2l.transformation

            map_pcd += pcd.transform(transform)

        map_pcd.normals = o3d.utility.Vector3dVector([])

        return map_pcd, poses

    else:

        map_pcd_ground = o3d.geometry.PointCloud()
        map_pcd_nonground = o3d.geometry.PointCloud()

        if ground_segmentation == 'patchwork':
            params = pypatchworkpp.Parameters()
            params.verbose = False
            PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        for i in tqdm(range(ind_start, ind_end)):
            panoptic_labels =  dataset[i].panoptic_labels
            semantic_labels =  dataset[i].semantic_labels
            instance_labels =  dataset[i].instance_labels
            

            
            pcd = get_pcd(dataset[i].point_cloud)
            pose = dataset.get_pose(i)
            poses.append(pose)
            transform = pose
            
            if ground_segmentation == 'patchwork':
                intensity = dataset[i].intensity
                PatchworkPLUSPLUS.estimateGround(np.hstack((np.asarray(pcd.points),intensity.reshape(-1,1))))
                ground_idcs = PatchworkPLUSPLUS.get_ground_idcs()
                nonground_idcs = PatchworkPLUSPLUS.get_nonground_idcs()
                

            elif ground_segmentation == 'open3d':
                _, ground_idcs = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=2000)
                nonground_idcs = np.setdiff1d(np.arange(0, np.asarray(pcd.points).shape[0]), ground_idcs)
            else:
                raise ValueError('ground_segmentation must be either "None", "patchwork" or "open3d"')
            
            
            
            panoptic_labels_nonground.append(panoptic_labels[nonground_idcs])
            panoptic_labels_ground.append(panoptic_labels[ground_idcs])
            
            seg_labels_nonground.append(semantic_labels[nonground_idcs])
            seg_labels_ground.append(semantic_labels[ground_idcs])
            
            instance_labels_ground.append(instance_labels[ground_idcs].copy())
            instance_labels_nonground.append(instance_labels[nonground_idcs].copy())
            
            pcd_ground = get_subpcd(pcd, ground_idcs) 
            pcd_nonground = get_subpcd(pcd, nonground_idcs)
            
            
            ###nuscenes store poses of ids
            if dataset.nuscenes : 
                transformed_pcd = o3d.geometry.PointCloud()
                transformed_pcd.points = pcd.points 
                colors = np.zeros((np.asarray(transformed_pcd.points).shape[0],3))
                transformed_pcd.transform(transform)
                pcd_points = np.asarray(transformed_pcd.points)
                unique_ids = np.unique(instance_labels)
                for ido in unique_ids: 
                    if ido in moving_ids or ido == 0: 
                        continue
                    
                    idcs = np.where(instance_labels == ido)[0] 
                    colors[idcs] = np.random.rand(3)
                    
                    cur_pts = pcd_points[idcs].copy()
                    inst_centroid = np.median(cur_pts,0)
                    if ido in pose_id_dict.keys():
                        
                        for past_pose in pose_id_dict[ido]:
                            dist = np.linalg.norm(past_pose - inst_centroid)
                            if dist > dataset.dist_threshold:
                                moving_ids.append(ido)
                                break 
                        pose_id_dict[ido].append(inst_centroid)
                        
                    else : 
                        pose_id_dict[ido]= [inst_centroid]
                
                #transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
                #o3d.visualization.draw_geometries([transformed_pcd])
                    
                
                pcd_grounds.append(pcd_ground.transform(transform))
                pcd_non_grounds.append(pcd_nonground.transform(transform))
            
            
        
            if icp:
                if i != ind_start:
                    merge = map_pcd_ground + map_pcd_nonground
                    merge = merge.voxel_down_sample(voxel_size=0.3)
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))

                    reg_p2l = registration.registration_icp(pcd, merge, icp_threshold, transform, registration.TransformationEstimationPointToPlane(), registration.ICPConvergenceCriteria(max_iteration=1000))
                    transform = reg_p2l.transformation

                pcd_ground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
                pcd_nonground.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=200))
            
            if dataset.nuscenes == False : 
                map_pcd_ground += pcd_ground.transform(transform)
                map_pcd_nonground += pcd_nonground.transform(transform)
                
        if dataset.nuscenes == True : 
            map_pcd_ground = o3d.geometry.PointCloud()
            map_pcd_nonground = o3d.geometry.PointCloud()
            
            nuscenes_panoptic_labels_nonground = []
            nuscenes_panoptic_labels_ground = []
            nuscenes_seg_labels_nonground = []
            nuscenes_seg_labels_ground = []
            nuscenes_instance_labels_ground = []
            nuscenes_instance_labels_nonground = []
            
            for i in range(len(pcd_grounds)):
                cur_panoptic_ground = panoptic_labels_ground[i]
                cur_panoptic_nonground = panoptic_labels_nonground[i]
                
                cur_instance_ground =    instance_labels_ground[i]
                cur_instance_nonground = instance_labels_nonground[i]
                
                cur_semantic_ground =    seg_labels_ground[i]
                cur_semantic_nonground = seg_labels_nonground[i]
                
                cur_pcd_ground = pcd_grounds[i]
                cur_pcd_nonground = pcd_non_grounds[i]
                moving_ids.sort()
                for ido in moving_ids : 
                    idcs_ground = np.where(cur_instance_ground.reshape(-1,) == ido)[0]
                    idcs_nonground = np.where(cur_instance_nonground.reshape(-1,) == ido)[0]
                    
                    if idcs_ground.shape[0] > 0 : 
                        cur_pcd_ground  = cur_pcd_ground.select_by_index(idcs_ground, invert=True)
                        cur_panoptic_ground = np.delete(cur_panoptic_ground,idcs_ground)
                        cur_semantic_ground = np.delete(cur_semantic_ground,idcs_ground)
                        cur_instance_ground = np.delete(cur_instance_ground,idcs_ground)
                    
                    if idcs_nonground.shape[0] > 0 : 
                        cur_pcd_nonground = cur_pcd_nonground.select_by_index(idcs_nonground,invert=True)
                        cur_panoptic_nonground = np.delete(cur_panoptic_nonground,idcs_nonground)
                        cur_semantic_nonground = np.delete(cur_semantic_nonground,idcs_nonground)
                        cur_instance_nonground = np.delete(cur_instance_nonground,idcs_nonground)
                 
                map_pcd_ground += cur_pcd_ground
                map_pcd_nonground += cur_pcd_nonground
                
                nuscenes_panoptic_labels_nonground.append(cur_panoptic_nonground.reshape(-1,1))
                nuscenes_panoptic_labels_ground.append(cur_panoptic_ground.reshape(-1,1))
                
                nuscenes_seg_labels_nonground.append(cur_semantic_nonground.reshape(-1,1))
                nuscenes_seg_labels_ground.append(cur_semantic_ground.reshape(-1,1))
                
                nuscenes_instance_labels_ground.append(cur_instance_ground.reshape(-1,1))
                nuscenes_instance_labels_nonground.append(cur_instance_nonground.reshape(-1,1))
                
            labels = {'seg_ground':nuscenes_seg_labels_ground,'seg_nonground':nuscenes_seg_labels_nonground,
                      'instance_ground':nuscenes_instance_labels_ground,'instance_nonground':nuscenes_instance_labels_nonground,
                      'panoptic_ground':nuscenes_panoptic_labels_ground,'panoptic_nonground':nuscenes_panoptic_labels_nonground}
                      

        
        else : 
            labels = {'seg_ground':seg_labels_ground,'seg_nonground':seg_labels_nonground,
                      'instance_ground':instance_labels_ground,'instance_nonground':instance_labels_nonground,
                      'panoptic_ground':panoptic_labels_ground,'panoptic_nonground':panoptic_labels_nonground}
                      
        map_pcd_ground.normals = o3d.utility.Vector3dVector([])
        map_pcd_nonground.normals = o3d.utility.Vector3dVector([])
            
        return map_pcd_ground, map_pcd_nonground, poses, world_pose, labels