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
        "-m", "--model_type",
        type=str,
        default="dinov2_vits14",
        help="""type of model to extract. 
                Choose from [dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14]"""
    )

    parser.add_argument(
        "-s", "--stride",
        type=int,
        default=7,
        help="""stride of first convolution layer. 
                small stride -> higher resolution."""
    )

    parser.add_argument(
        "--facet",
        type=str,
        default="token",
        help="""facet of feature map to extract. 
                options: ['key' | 'query' | 'value' | 'token']"""
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    Adapter(args.image_path, args.image_format, args.output_path, args.model_type, args.stride, args.facet).run()