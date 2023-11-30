import numpy as np
import open3d as o3d
import os
import struct
from points_removal_scripts.mesh_based_script.mesh_generation import generate_mesh
import igl

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

def triangle_center(v, f):
    return (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3

def hidden_points_removal(point_cloud_path, view_point, threshold, mesh = None):
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

    # Creating rays from view point to points with normalization
    pcd_points = np.asarray(pcd.points)
    direction_vectors = pcd_points - view_point
    distances_to_points = np.linalg.norm(direction_vectors, axis=1)
    rays = np.zeros((len(pcd_points), 6))
    rays[:, :3] = view_point
    rays[:, 3:] = direction_vectors / distances_to_points[:, None]
    rays = rays.astype(np.float32)
    cast_results = scene.cast_rays(rays)

    # Selecting points that are no more than threshold away from the intersection with the mesh
    hit_distances = cast_results["t_hit"].numpy()
    visibility_mask = (hit_distances + threshold) >= distances_to_points
    estimated_visibility = visibility_mask

    pcd_points_new = pcd_points[visibility_mask]
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_new))
    return new_pcd, estimated_visibility

def hidden_points_removal_rt_mesh(point_cloud_path, view_point, threshold, mesh = None):
    ext = os.path.splitext(point_cloud_path)[-1]
    if ext == ".bin":
        pcd = read_bin_file(point_cloud_path)
    else:
        pcd = o3d.io.read_point_cloud(point_cloud_path)
    if not mesh:
        mesh = generate_mesh(point_cloud_path)
    
    points = np.asarray(pcd.points)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    face_centers = triangle_center(vertices, faces)

    # calculate the association array of point and triangle face
    _, point_face_idx, _ = igl.point_mesh_squared_distance(points, vertices, faces)
    direction_vec = face_centers - view_point
    distance = np.linalg.norm(direction_vec, axis=1)

    # ray casting over face centers
    scene = o3d.t.geometry.RaycastingScene()
    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh1)

    rays = np.zeros((face_centers.shape[0], 6))
    rays[:, :3] = view_point
    rays[:, 3:] = direction_vec / distance[:, None]
    rays = rays.astype(np.float32)
    cast_results = scene.cast_rays(rays)

    t_hit = cast_results["t_hit"].numpy()
    visibilty_mask_faces = distance - t_hit <= threshold

    # convert face visibility to point visibility
    point_mask = visibilty_mask_faces[point_face_idx]
    pcd_points_new = points[point_mask]
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_new))

    return new_pcd, point_mask