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
        "-n", "--n_segments",
        type=int,
        default=100,
        help="numbers of estimated segments"
    )

    parser.add_argument(
        "-m", "--mslic",
        type=bool,
        default=True,
        help="use mask slic if true"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    Adapter(args.image_path, args.image_format, args.output_path, args.n_segments, args.mslic).run()