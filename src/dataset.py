import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

class RandomNoise(object):
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, features):
        noise = np.random.normal(0, self.noise_level, features.shape)
        return features + noise

class CircuitNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.label_map = {}

        # Load all .npz files in the directory
        files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        for file in tqdm(files, desc="Loading files"):
            file_path = os.path.join(data_dir, file)
            self.load_file(file_path)

    def load_file(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        pin_positions = data['pin_positions'].item()

        for key, pin_data in pin_positions.items():
            x1, y1, x2, y2, gx1, gy1, gx2, gy2 = pin_data
            features = np.array([x1, y1, x2, y2, gx1, gy1, gx2, gy2, x2-x1, y2-y1, (x2-x1)*(y2-y1)], dtype=np.float32)
            component_type = key.split('/')[0]

            if component_type not in self.label_map:
                self.label_map[component_type] = len(self.label_map)

            label = self.label_map[component_type]
            self.samples.append((features, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), label

    def get_label_distribution(self):
        labels = [sample[1] for sample in self.samples]
        return Counter(labels)

def create_data_loaders(dataset, batch_size, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader