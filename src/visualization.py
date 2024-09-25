import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_attention(model, dataloader, device, num_samples=5, save_path='attention_visualization.png'):
    model.eval()
    fig, axes = plt.subplots(num_samples, len(model.transformer_layers), figsize=(5*len(model.transformer_layers), 5*num_samples))
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            _, attention_weights = model(inputs)
            
            for j, attn_weights in enumerate(attention_weights):
                attn = attn_weights[0].squeeze().cpu()  # Get the first sample in the batch
                if num_samples == 1 and len(model.transformer_layers) == 1:
                    ax = axes
                elif num_samples == 1:
                    ax = axes[j]
                elif len(model.transformer_layers) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                sns.heatmap(attn, ax=ax, cmap='viridis')
                ax.set_title(f"Sample {i+1}, Layer {j+1}")
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention visualization saved to {save_path}")

def plot_training_history(train_losses, val_losses, accuracies, save_path='training_history.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

# Example usage
if __name__ == "__main__":
    from dataset import CircuitNetDataset, create_data_loaders
    from model import CircuitFormer
    import torch.nn as nn

    RUN_EXAMPLE = os.environ.get('RUN_EXAMPLES', 'False').lower() == 'true'

    if RUN_EXAMPLE:
        print("Running visualization examples...")
        
        # Setup
        data_dir = r'C:\Users\nihar\Documents\github\vlsi_congestion_predictor\data\pin_positions'
        dataset = CircuitNetDataset(data_dir)
        _, _, test_loader = create_data_loaders(dataset, batch_size=64)

        # Model parameters
        input_dim = 11
        hidden_dim = 256
        num_layers = 6
        num_heads = 8
        max_seq_length = 1
        num_classes = len(dataset.label_map)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CircuitFormer(input_dim, hidden_dim, num_layers, num_heads, max_seq_length, num_classes).to(device)

        # Load trained model weights (assuming you have a trained model)
        model.load_state_dict(torch.load('circuitformer_model.pth'))

        # Visualize attention
        visualize_attention(model, test_loader, device, num_samples=3)

        # Plot training history (with dummy data for demonstration)
        train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        val_losses = [0.55, 0.45, 0.35, 0.25, 0.15]
        accuracies = [0.6, 0.7, 0.8, 0.85, 0.9]
        plot_training_history(train_losses, val_losses, accuracies)

        print("Visualization examples completed.")
    else:
        print("Visualization example code is available but not running.")
        print("Set the RUN_EXAMPLES environment variable to 'True' to run the examples.")