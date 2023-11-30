import argparse
import ast
import glob
import os
import warnings

import numpy as np
from open3d.io import write_point_cloud, write_triangle_mesh
from points_removal_scripts.mesh_based_script import (HPR_mesh_based,
                                                      mesh_generation)


warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'true', '1'):
        return True
    elif v.lower() in ('False', 'false', '0'):
        return False

def save_mask(mask, filename):
    np.savez_compressed(filename, mask = mask)

def save_point_cloud(pcd, filename):
    write_point_cloud(filename, pcd)

def save_results(center_points, pt_masks, filename):
    np.savez_compressed(filename, center_points=center_points, pt_masks=pt_masks)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="""
        This script takes path to the cloud data
        """
    )

    parser.add_argument("--point_cloud_path", default="/workspace/dataset" , help="path to cloud data")
    parser.add_argument("--center_point", type=str, default="[0, 0, 0]", help="center point")
    parser.add_argument("--center_point_file", default=None, help="file name of center point file")
    parser.add_argument("--output_path", default="/workspace/output", help="path to output file")
    parser.add_argument("--point_cloud_output", default=False, help="if you want to save masked point cloud")
    parser.add_argument("--threshold", type=float, default=0.1, help="threshold")
    parser.add_argument("--mesh_output", default=False, help="if you want to save mesh")
    parser.add_argument("--cast_on_mesh", default=True, help="if you want to cast rays on mesh")

    args = parser.parse_args()
    
    if args.point_cloud_path.endswith(".pcd") and args.point_cloud_path.endswith(".bin") and args.point_cloud_path.endswith(".ply"):
        pcd_path = [args.point_cloud_path]
    else:
        pcd_path = glob.glob(args.point_cloud_path + "/*.pcd")
        pcd_path.extend(glob.glob(args.point_cloud_path + "/*.bin"))
        pcd_path.extend(glob.glob(args.point_cloud_path + "/*.ply"))
    
    print(f"Found {len(pcd_path)} point clouds")
    for pcd in pcd_path:
        print(f"Processing {pcd}")
        pcd_name = os.path.splitext(os.path.basename(pcd))[0]

        pt_masks = [] # list of masks
        pcds_out = [] # list of masked point clouds
        print("Generating mesh...")
        mesh = mesh_generation.generate_mesh(pcd)
        print("Mesh generation done")

        if args.center_point_file: # multiple center points
            print(f"Loading center points from /workspace/dataset/{args.center_point_file}")
            center_points = np.load(f"/workspace/dataset/{args.center_point_file}")
        else:
            center_points = [ast.literal_eval(args.center_point)] # single center point
        print(f"Perform HPR for {len(center_points)} center points")

        for center_point in center_points: 
            if str2bool(args.cast_on_mesh):
                masked_pcd, pt_map = HPR_mesh_based.hidden_points_removal_rt_mesh(pcd, center_point, args.threshold, mesh)
            else:
                masked_pcd, pt_map = HPR_mesh_based.hidden_points_removal(pcd, center_point, args.threshold, mesh)

            pt_masks.append(pt_map)
            pcds_out.append(masked_pcd)

        if os.path.exists(args.output_path) == False:
            os.mkdir(args.output_path)

        save_results(center_points, pt_masks, args.output_path + f"/{pcd_name}_masks_{args.threshold}.npz")

        if str2bool(args.point_cloud_output):
            for i, pcd_out in enumerate(pcds_out):
                write_point_cloud(args.output_path + f"/{pcd_name}_{i}_pcd_{args.threshold}.ply", pcd_out)

        if str2bool(args.mesh_output):
            write_triangle_mesh(args.output_path + f"/{pcd_name}_mesh.ply", mesh)

