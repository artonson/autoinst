import glob
import os

import numpy as np
from PIL import Image
from skimage import color, morphology, segmentation
from skimage.filters import threshold_otsu
from tqdm import tqdm


class Adapter():
    def __init__(
            self,
            image_path: str,
            image_format: str,
            output_path: str,
            n_segments: int,
            mslic: bool,
            ) -> None:
        self.image_path = glob.glob(image_path + "/*." + image_format)
        self.output_path = output_path
        self.n_segments = n_segments
        self.mslic = mslic



    def _generate_mask(self, image: np.ndarray) -> np.ndarray:
        gray = color.rgb2gray(image)
        thresh = threshold_otsu(gray)
        mask = morphology.remove_small_holes(
            morphology.remove_small_objects(    
                gray < thresh, 500),
            500)
        return mask
    

    def _get_segmentation(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.mslic:
            return segmentation.slic(image, n_segments=self.n_segments, mask=mask)
        else:
            return segmentation.slic(image, n_segments=self.n_segments)
        
    def _segmentation_to_mask(self, segmentation: np.ndarray) -> list:
        masks = []
        unique_ids = np.unique(segmentation)
        
        for unique_id in unique_ids:
            mask = (segmentation == unique_id)
            masks.append(mask)
        
        return masks

    def _calculate_bbox_and_area(self, mask: np.ndarray) -> tuple:
        true_coords = np.transpose(np.where(mask))
        min_coords = np.min(true_coords, axis=0)
        max_coords = np.max(true_coords, axis=0)
        x = min_coords[1]
        y = min_coords[0]
        w = max_coords[1] - min_coords[1] + 1
        h = max_coords[0] - min_coords[0] + 1
        bbox = [x, y, w, h]
        area = np.sum(mask)
        
        return bbox, area
    

    def _post_process_segmentation(self, segmentation: np.ndarray) -> list:
        masks = self._segmentation_to_mask(segmentation)
        processed_segs = []
        for mask in masks:
            bbox, area = self._calculate_bbox_and_area(mask)
            processed_seg = {
                "segmentation": mask,
                "bbox": bbox,
                "area": area
            }
            processed_segs.append(processed_seg)
        return processed_segs
    
    
    def _save_prediction(self, filename, masks: list) -> None:
        np.savez_compressed(filename, masks = masks)


    def run(self) -> None:
        pbar = tqdm(self.image_path, total=len(self.image_path))
        for image_path in pbar:
            image = np.asarray(Image.open(image_path))
            filename = os.path.join(self.output_path, os.path.basename(image_path).split(".")[0] + ".npz")
            pbar.set_description(f'Processing {image_path}, Saving prediction to {filename}')
            mask = self._generate_mask(image)
            segmentation = self._get_segmentation(image, mask)
            processed_segs = self._post_process_segmentation(segmentation)
            self._save_prediction(filename, processed_segs)