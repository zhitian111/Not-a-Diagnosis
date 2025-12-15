# /*
#  * Copyright (c) 2025 zhitian111
#  * Released under the MIT license. See LICENSE for details.
#  */
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LIDCNoduleDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir/
            ├── benign/
            │   ├── xxx.npy
            └── malignant/
                ├── yyy.npy
        """
        self.samples = []

        for label_name, label in [("benign", 0), ("malignant", 1)]:
            class_dir = os.path.join(root_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        patch = np.load(path, mmap_mode="r")  # (64, 64, 64)
        # patch = np.asarray(patch, dtype=np.float32)
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)

        label = torch.tensor(label, dtype=torch.long)

        x = patch.float()

        return x, label
