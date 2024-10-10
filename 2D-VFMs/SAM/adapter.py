import glob
import os

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


class Adapter():
    def __init__(
            self,
            image_path: str,
            image_format: str,
            output_path: str,
            model_path: str,
            ) -> None:
        self.image_path = glob.glob(image_path + "/*." + image_format)
        self.output_path = output_path
        self.model_path = model_path
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = "cuda"
        else:
            print("Using CPU")
            self.device = "cpu"


    def _build_model(self) -> SamAutomaticMaskGenerator:
        sam = sam_model_registry["default"](checkpoint = self.model_path)
        sam = sam.to(self.device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator
    

    def _predict(self, image: Image, mask_generator: torch.nn.Module) -> list:
        masks = mask_generator.generate(image)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        return sorted_masks
    

    def _post_process_prediction(self, masks: list) -> list:
        processed_masks = []
        for mask in masks:
            processed_mask = {key: mask[key] for key in ["segmentation", "bbox", "area", "predicted_iou", "stability_score"]}
            processed_masks.append(processed_mask)
        return processed_masks
    
    
    def _save_prediction(self, filename, masks: list) -> None:
        np.savez_compressed(filename, masks = masks)


    def run(self) -> None:
        mask_generator = self._build_model()
        pbar = tqdm(self.image_path, total=len(self.image_path))
        for image_path in pbar:
            image = np.asarray(Image.open(image_path))
            filename = os.path.join(self.output_path, os.path.basename(image_path).split(".")[0] + ".npz")
            if os.path.exists(filename) == True : 
                print("output sam file already exists, skipping") 
                continue 
            pbar.set_description(f'Processing {image_path}, Saving prediction to {filename}')
            masks = self._predict(image, mask_generator)
            processed_masks = self._post_process_prediction(masks)
            self._save_prediction(filename, processed_masks)
