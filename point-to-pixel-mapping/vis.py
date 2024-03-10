import open3d as o3d
import numpy as np
import copy
from functools import partial
from dataset_utils import *


class Counter:
    def __init__(self, pcd_base) -> None:
        self.counter = 0
        self.pcd = pcd_base
        self.pcd_orig = copy.deepcopy(pcd_base)
        self.idcs = "all"
        self.img_cnt = {i: 0 for i in range(len(pcd_arr) + 1)}
        self.current_idx = 0
        self.poses = []
        self.pose_idx = -1

    def find_closest_pose(self, center):
        closest = None
        closest_dist = 10000
        pose_idx = -1

        for entry in self.poses:
            pose = entry[0]
            cur_dist = np.linalg.norm(np.array(pose[:2]) - np.array(center))
            if cur_dist < closest_dist:
                closest = pose
                closest_dist = cur_dist
                pose_idx = entry[1]
        return closest, pose_idx


folder_base = "/home/cedric/unsup_3d_instances/point-to-pixel-mapping/pcd_preprocessed/"

# fn = "002664.pcd"
##name_refined = folder_base + fn
# name_base = folder_base + "similarity_" + fn
name_refined = folder_base + "merge_refined5.pcd"
name_base = folder_base + "tarl_dino_spatial55.pcd"
name_raw = "chunk_cur_all.pcd"


pcd_refined = o3d.io.read_point_cloud(name_refined)
pcd_base = o3d.io.read_point_cloud(name_base)
pcd_raw = o3d.io.read_point_cloud(name_raw)


cols_refined = pcd_refined.colors
cols_base = pcd_base.colors

pcd_arr = [copy.deepcopy(pcd_refined), copy.deepcopy(pcd_base), copy.deepcopy(pcd_raw)]

DATASET_PATH = os.path.join("/media/cedric/Datasets2/semantic_kitti/")
SEQUENCE_NUM = 5
dataset = create_kitti_odometry_dataset(DATASET_PATH, SEQUENCE_NUM, ncuts_mode=True)
all_poses = []
cnt = Counter(pcd_base)
for i in range(0, len(dataset), 1):
    cur_pose = dataset.get_pose(i)
    T = cur_pose[:3, -1]
    all_poses.append([T, i])
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    # sphere.paint_uniform_color([1, 0, 0])  # Red color
    # sphere.translate([T[0],T[1],T[2]])  # Move the sphere to the specified center
    # vis.add_geometry(sphere)

cnt.poses = all_poses
cur_pcd = o3d.geometry.PointCloud()
cur_pcd.points = o3d.utility.Vector3dVector(dataset.get_point_cloud(2660))
# o3d.visualization.draw_geometries([cur_pcd])


# Function to change the colors of the point cloud
def change_point_cloud_color(pcd, idx):
    # Change the color of all points in the point cloud
    if cnt.idcs == "all":
        if idx < 3:
            pcd.colors = pcd_arr[idx].colors
        elif idx == 3:
            points = np.asarray(pcd.points)
            # Normalize Z values to 0-1 range for color mapping
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            z_min = -2.40
            colors = (points[:, 2] - z_min) / (z_max - z_min)
            print(z_min)
            print(z_max)
            # Apply colormap (can choose any available colormap in matplotlib)
            colors = plt.get_cmap("viridis")(colors)[
                :, :3
            ]  # Use the first three columns to ignore alpha channel

            # Assign colors back to PCD
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud("chunk_all5.pcd", pcd)
    else:
        if idx < 3:
            subpcd = pcd_arr[idx].select_by_index(cnt.idcs)
            pcd.colors = subpcd.colors
        elif idx == 3:
            points = np.asarray(pcd.points)
            # Normalize Z values to 0-1 range for color mapping
            print(z_min)
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            colors = (points[:, 2] - z_min) / (z_max - z_min)
            print(z_max)
            # Apply colormap (can choose any available colormap in matplotlib)
            colors = plt.get_cmap("viridis")(colors)[
                :, :3
            ]  # Use the first three columns to ignore alpha channel

            # Assign colors back to PCD
            pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# Visualize the point cloud
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Point Cloud Visualization")


def key_callback(vis, key_code):
    if key_code == 49:  # Right arrow key
        idx = 0  # Red
        cnt.current_idx = idx
    elif key_code == 50:  # Default color
        idx = 1  # Blue
        cnt.current_idx = idx
    elif key_code == 51:
        idx = 2
        cnt.current_idx = idx
    else:
        idx = 3
        cnt.current_idx = idx

    change_point_cloud_color(cnt.pcd, idx)
    # print(np.asarray(cur_pcd.colors)[:10])
    vis.update_geometry(cnt.pcd)
    vis.poll_events()
    vis.update_renderer()


def crop_point_cloud(pcd_base, center, param, extent=25):

    # if center[0] == 69.28762817382812 :
    #    return

    # Create a bounding box with the specified extent centered at the specified position
    max_position = center + (0.5 * extent)
    min_position = center - (0.5 * extent)
    points = np.asarray(pcd_base.points)
    ids = np.where(
        np.all(points[:, :2] > min_position, axis=1)
        & np.all(points[:, :2] < max_position, axis=1)
    )[0]

    new_pcd = pcd_base.select_by_index(ids)
    # Crop the point cloud
    # o3d.visualization.draw_geometries([pcd_base])
    # vis.remove_geometry(pcd_base)

    # vis.clear_geometry()
    ctr = vis.get_view_control()
    vis.add_geometry(new_pcd.paint_uniform_color([1, 0, 0]))
    vis.poll_events()
    # ctr.convert_from_pinhole_camera_parameters(param.extrinsic)
    vis.update_renderer()


def mouse_move_callback(vis):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    x = param.extrinsic[0][-1]
    y = param.extrinsic[1][-1]
    z = param.extrinsic[2][-1]
    new_param = copy.deepcopy(param)
    center_mouse = np.array([-x, y])
    ##find closest pose to it
    extent = 25
    center, pose_idx = cnt.find_closest_pose(center_mouse)
    print("current pose idx ", pose_idx)
    cnt.pose_idx = pose_idx
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)
    # sphere.paint_uniform_color([0, 1, 0])  # Red color
    # sphere.translate([center[0],center[1],center[2]])
    # vis.add_geometry(sphere)
    center = center[:2]

    max_position = center + (0.5 * extent)
    min_position = center - (0.5 * extent)
    points = np.asarray(cnt.pcd_orig.points)
    ids = np.where(
        np.all(points[:, :2] > min_position, axis=1)
        & np.all(points[:, :2] < max_position, axis=1)
    )[0]
    cnt.idcs = ids
    current_pcd = cnt.pcd_orig.select_by_index(ids)

    points = np.asarray(current_pcd.points)
    # Normalize Z values to 0-1 range for color mapping
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    # z_min = -2.40
    colors = (points[:, 2] - z_min) / (z_max - z_min)
    print(z_min)
    print(z_max)
    # Apply colormap (can choose any available colormap in matplotlib)
    colors = plt.get_cmap("viridis")(colors)[
        :, :3
    ]  # Use the first three columns to ignore alpha channel

    # Assign colors back to PCD
    current_pcd.colors = o3d.utility.Vector3dVector(colors)
    cols_arr = np.asarray(cnt.pcd.colors)
    cols_arr[ids] = colors
    cnt.pcd.colors = o3d.utility.Vector3dVector(cols_arr)
    o3d.visualization.draw_geometries([cnt.pcd])

    o3d.io.write_point_cloud("chunk_cur5.pcd", cnt.pcd)
    # o3d.io.write_point_cloud("chunk_cur.pcd", current_pcd)
    # cnt.pcd = cnt.pcd_orig.select_by_index(ids)
    vis.clear_geometries()
    vis.add_geometry(cnt.pcd)
    vis.update_renderer()


def button_forward(vis, extent=25):
    center = np.array(all_poses[cnt.counter])[:2]
    cnt.counter += 2
    max_position = center + (0.5 * extent)
    min_position = center - (0.5 * extent)
    points = np.asarray(cnt.pcd_orig.points)
    ids = np.where(
        np.all(points[:, :2] > min_position, axis=1)
        & np.all(points[:, :2] < max_position, axis=1)
    )[0]
    cnt.idcs = ids
    cnt.pcd = cnt.pcd_orig.select_by_index(ids)
    vis.clear_geometries()
    vis.add_geometry(cnt.pcd)
    # o3d.visualization.draw_geometries([cnt.pcd])


def take_screenshot(vis):
    img_name = (
        "image_center_idx"
        + str(cnt.pose_idx)
        + "_current_idx_"
        + str(cnt.current_idx)
        + "_"
        + str(cnt.img_cnt[cnt.current_idx])
        + ".png"
    )
    vis.capture_screen_image("screenshots/" + img_name)


def reset_geometry(vis):
    cnt.idcs = "all"
    vis.clear_geometries()
    cnt.pcd = cnt.pcd_orig
    vis.add_geometry(cnt.pcd)


vis.register_key_callback(ord("R"), partial(reset_geometry))
vis.register_key_callback(ord("S"), partial(take_screenshot))
vis.register_key_callback(ord("M"), partial(mouse_move_callback))
vis.register_key_callback(50, partial(key_callback, key_code=50))  # Right arrow key
vis.register_key_callback(52, partial(key_callback, key_code=52))  # Right arrow key
vis.register_key_callback(51, partial(key_callback, key_code=51))  # Right arrow key
vis.register_key_callback(49, partial(key_callback, key_code=49))  # Left arrow key


vis.add_geometry(pcd_base)
vis.run()
# print(vis.get_picked_points())
vis.destroy_window()
