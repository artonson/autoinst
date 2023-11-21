import numpy as np
import open3d as o3d
import os
import struct
from points_removal_scripts.mesh_based_script.mesh_generation import generate_mesh

def read_bin_file(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def hidden_points_removal(point_cloud_path, center_point, threshold, mesh = None):
    ext = os.path.splitext(point_cloud_path)[-1]
    if ext == ".bin":
        pcd = read_bin_file(point_cloud_path)
    else:
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    if mesh:
        mesh_0 = mesh
    else:
        mesh_0 = generate_mesh(point_cloud_path)
    # o3d.io.write_triangle_mesh("/workspace/output/mesh.ply", mesh)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_0)
    # Creating raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    # Creating rays from (0, 0, 0) to points with normalization
    pcd_points = np.asarray(pcd.points)
    direction_vectors = pcd_points - center_point
    distances_to_points = np.linalg.norm(direction_vectors, axis=1)
    rays = np.zeros((len(pcd_points), 6))
    rays[:, :3] = center_point
    rays[:, 3:] = direction_vectors
    rays = rays / distances_to_points[:, None]
    rays = rays.astype(np.float32)
    cast_results = scene.cast_rays(rays)

    # Selecting points that are no more than 10 cm away from the intersection with the mesh
    hit_distances = cast_results["t_hit"].numpy()
    visibility_mask = (hit_distances + threshold) >= distances_to_points
    estimated_visibility = visibility_mask

    pcd_points_new = pcd_points[visibility_mask]
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_new))
    return new_pcd, estimated_visibility
