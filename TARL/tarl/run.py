
import os
from argparse import ArgumentParser

from tarl_extractor import TARLFOLDER

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_path",
        type=str, 
        default="/input", 
        help="path to input directory (where pointclouds are stored)"
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str, 
        default="kitti", 
        help="dataset format, either kitti or nuscenes is supported"
    )
    
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="/output",
        help="path to output directory"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    tarl_folder = TARLFOLDER(args)
    tarl_folder.run_on_folder()