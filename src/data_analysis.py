from dataset import CircuitNetDataset, RandomNoise, create_data_loaders
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

def analyze_dataset(dataset):
    print(f"Total dataset size: {len(dataset)}")

    # Print label distribution
    label_dist = dataset.get_label_distribution()
    print("Label distribution:")
    for label, count in label_dist.most_common():
        print(f"Label {label}: {count} samples")

    # Print feature statistics
    print("Calculating feature statistics...")
    feature_sum = torch.zeros(11)
    feature_sq_sum = torch.zeros(11)
    feature_min = torch.full((11,), float('inf'))
    feature_max = torch.full((11,), float('-inf'))
    for i in tqdm(range(len(dataset)), desc="Processing features"):
        features, _ = dataset[i]
        feature_sum += features
        feature_sq_sum += features ** 2
        feature_min = torch.min(feature_min, features)
        feature_max = torch.max(feature_max, features)

    n = len(dataset)
    feature_mean = feature_sum / n
    feature_std = torch.sqrt(feature_sq_sum / n - feature_mean ** 2)

    print(f"\nFeature statistics:")
    print(f"Mean: {feature_mean}")
    print(f"Std: {feature_std}")
    print(f"Min: {feature_min}")
    print(f"Max: {feature_max}")

    # Visualize data
    print("Generating scatter plot...")
    plt.figure(figsize=(10, 10))
    sample_size = min(10000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    sampled_features = torch.stack([dataset[i][0] for i in indices])
    plt.scatter(sampled_features[:, 0], sampled_features[:, 1], alpha=0.1)
    plt.title("Component Positions (Sampled)")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.savefig("component_positions.png")
    plt.close()

# Usage
data_dir = r'C:\Users\nihar\Documents\github\vlsi_congestion_predictor\data\pin_positions'
transform = RandomNoise(noise_level=0.005)
dataset = CircuitNetDataset(data_dir, transform=transform)

# Analyze the dataset
analyze_dataset(dataset)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=64)

# Print information about the splits
print(f"Train set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(val_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")

# Print first batch from train_loader
print("Fetching first batch from train_loader...")
for batch_features, batch_labels in train_loader:
    print(f"\nBatch shape: {batch_features.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    print(f"First few features in this batch:\n{batch_features[:5]}")
    print(f"First few labels in this batch: {batch_labels[:5]}")
    break