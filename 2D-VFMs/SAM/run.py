import os
from argparse import ArgumentParser

from adapter import Adapter

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--image_path",
        type=str, 
        default="/input", 
        help="path to image"
    )

    parser.add_argument(
        "-f", "--image_format",
        type=str, 
        default="png", 
        help="image format, default is png"
    )
    
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        default="/output",
        help="path to output directory"
    )

    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="checkpoints/sam_vit_h_4b8939.pth",
        help="path to SAM checkpoint"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    Adapter(args.image_path, args.image_format, args.output_path, args.model_path).run()