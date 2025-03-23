import h5py
import numpy as np
import os
from pathlib import Path

class HDF5FeatureWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.file = None
        self.dataset = None
        
    def __enter__(self):
        self.file = h5py.File(self.path, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"Features saved to: {self.path.resolve()}")
            
    def write_batch(self, features: np.ndarray):
        if features.size == 0:
            return
            
        if not self.dataset:
            self._create_dataset(features.shape[1])
            
        self.dataset.resize(self.dataset.shape[0] + features.shape[0], axis=0)
        self.dataset[-features.shape[0]:] = features
        
    def _create_dataset(self, feat_dim: int):
        self.dataset = self.file.create_dataset(
            "features",
            shape=(0, feat_dim),
            maxshape=(None, feat_dim),
            chunks=(512, feat_dim),
            dtype=np.float32
        )