import open3d as o3d
import os
from dataset_utils import create_kitti_odometry_dataset
import numpy as np
import copy
from tqdm import tqdm
import time
from scipy.spatial.transform import Rotation as R


DATASET_PATH = os.path.join("/media/cedric/Datasets2/semantic_kitti/")
# pcd_path = "pcd_preprocessed/ncuts_tarl_spatial1_all.pcd"
seqs = [0]
# seqs = [6, 8, 9, 10]
print("Seqs", seqs)
div = 0
interpolation_steps = 3

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=3000, height=2000, visible=True)


names = [
    "spatial_1.0_tarl_0.5_t_0.03",
    "sam3d_fix",
    "maskpls_tarl_spatial_",
    "3duis_fix",
    "hdbscan",
    "kitti_panoptic_",
]


for seq in tqdm(seqs, desc="Sequence"):
    print("Seq", seq)
    folder_base = f"/media/cedric/Datasets21/Semantics/KITTI/{seq}/"
    dataset = create_kitti_odometry_dataset(DATASET_PATH, seq, ncuts_mode=True)
    for cur_name in names:
        print("cur name", cur_name)
        name = f"{cur_name}{seq}_{div}.pcd"
        pcd_path = folder_base + name

        pcd = o3d.io.read_point_cloud(pcd_path)
        o3d.io.write_point_cloud(pcd_path.replace("pcd", "ply"), pcd)
        continue

        out_pth = f"screenshots/{seq}/{div}/{name.split('.pcd')[0]}_update_test"
        if os.path.exists(out_pth) == False:
            os.makedirs(out_pth)
        cnt = 0

        test_idcs = list(range(30, 900 - 1))

        for test_idx in tqdm(test_idcs):
            prev_pose = dataset.get_pose(test_idx)
            next_pose = dataset.get_pose(test_idx + 1)
            for i in range(0, interpolation_steps):
                cur_pose = prev_pose * (
                    (interpolation_steps - i) / interpolation_steps
                ) + next_pose * (i / interpolation_steps)

                cur_pose[2, -1] += 5.0
                T_world2lidar = np.linalg.inv(cur_pose)
                T_lidar2cam, K = dataset.get_calibration_matrices("cam2")
                T_world2cam = T_lidar2cam @ T_world2lidar
                T_pcd2cam = T_world2cam @ dataset.get_pose(0)

                vis.add_geometry(pcd)
                ctr = vis.get_view_control()

                camera_params = ctr.convert_to_pinhole_camera_parameters()
                # cur_pose[:-1, :-1] = np.identity(3)
                # T_pcd2cam[1, -1] += 2.0

                camera_params.extrinsic = T_pcd2cam
                intrinsic = dataset.get_calibration_matrices("cam2")[1]
                camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=3000, height=2000, intrinsic_matrix=intrinsic
                )
                rot = np.eye(4)
                rot[:3, :3] = R.from_euler("x", 57, degrees=True).as_matrix()
                rot = rot.dot(camera_params.extrinsic)
                camera_params.extrinsic = rot

                ctr.convert_from_pinhole_camera_parameters(camera_params, True)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(1)

                vis.capture_screen_image(f"{out_pth}/{str(cnt).zfill(6)}.png")

                cnt += 1

                # vis.run()
                vis.remove_geometry(pcd)

vis.destroy_window()
