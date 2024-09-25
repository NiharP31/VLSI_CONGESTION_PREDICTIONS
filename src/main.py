import torch
import torch.nn as nn
from dataset import CircuitNetDataset, create_data_loaders, RandomNoise
from model import CircuitFormer
from train import train_circuitformer
from evaluation import evaluate_model, plot_confusion_matrix
from visualization import visualize_attention, plot_training_history

def main():
    # Hyperparameters
    data_dir = r'C:\Users\nihar\Documents\github\vlsi_congestion_predictor\data\pin_positions'
    input_dim = 11
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    max_seq_length = 1
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001

    # Load and prepare data
    transform = RandomNoise(noise_level=0.005)
    dataset = CircuitNetDataset(data_dir, transform=transform)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=batch_size)

    num_classes = len(dataset.label_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = CircuitFormer(input_dim, hidden_dim, num_layers, num_heads, max_seq_length, num_classes).to(device).to(torch.float32)

    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses, accuracies = train_circuitformer(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Plot training history
    plot_training_history(train_losses, val_losses, accuracies)

    # Evaluate model
    test_loss, test_accuracy, all_predictions, all_targets = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot confusion matrix
    class_names = list(dataset.label_map.keys())
    plot_confusion_matrix(all_targets, all_predictions, class_names)

    # Visualize attention
    visualize_attention(model, test_loader, device)

    # Save the model
    model_save_path = 'circuitformer_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    main()