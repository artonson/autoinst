import glob
import os

import numpy as np
import torch
from extractor import ViTExtractor
from torch import nn
from tqdm import tqdm


class Adapter():
    def __init__(
            self,
            image_path: str,
            image_format: str,
            output_path: str,
            model_type: str,
            stride: int = 7,
            facet: str = "token",
            layer: int = 11
            ) -> None:
        if not image_path.endswith(image_format):
            self.image_path = glob.glob(image_path + "/*." + image_format)
        else:
            self.image_path = [image_path]
        self.output_path = output_path
        self.model_type = model_type
        self.stride = stride
        self.facet = facet
        self.layer = layer
        if torch.cuda.is_available():
            print("Using GPU")
            self.device = "cuda"
        else:
            print("Using CPU")
            self.device = "cpu"


    def _build_model(self) -> nn.Module:
        model = ViTExtractor(self.model_type, self.stride, model= None, device = self.device)
        self.feature_dim = model.model.num_features
        self.max_layer = model.model.n_blocks - 1
        assert self.layer <= self.max_layer, f"Layer {self.layer} does not exist. Please choose a layer between 0 and {self.max_layer}"
        return model
    

    def _extract_features(self, image_path: str, model: ViTExtractor) -> np.ndarray:
        image_batch, _ = model.preprocess(image_path)
        features = model.extract_descriptors(image_batch.to(self.device), layer=self.layer, facet=self.facet, bin=False)
        self.num_patches = model.num_patches
        pre_features = self._post_process_features(features)
        return pre_features
    

    def _post_process_features(self, features: list) -> np.ndarray:
        features = features.cpu().numpy()
        pre_features = features.reshape((self.num_patches[0], self.num_patches[1], self.feature_dim))
        return pre_features


    def _save_features(self, filename, feature_map: np.ndarray) -> None:
        np.savez_compressed(filename, feature_map = feature_map)


    def run(self) -> None:
        with torch.no_grad():
            extractor = self._build_model()
            pbar = tqdm(self.image_path, total=len(self.image_path))
            for image_path in pbar:
                filename = os.path.join(self.output_path, os.path.basename(image_path).split(".")[0] + ".npz")
                pbar.set_description(f'Processing {image_path}, Saving prediction to {filename}')
                features = self._extract_features(image_path, extractor)
                self._save_features(filename, features)
