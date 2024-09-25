import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)  # Unpack the tuple, ignore attention weights
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))

    return avg_loss, accuracy, all_predictions, all_targets

def plot_confusion_matrix(all_targets, all_predictions, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

# Example usage
if __name__ == "__main__":
    from dataset import CircuitNetDataset, create_data_loaders
    from model import CircuitFormer
    import torch.nn as nn

    RUN_EXAMPLE = os.environ.get('RUN_EXAMPLES', 'False').lower() == 'true'

    if RUN_EXAMPLE:
        print("Running example evaluation...")
        
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

        # Initialize model and criterion
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CircuitFormer(input_dim, hidden_dim, num_layers, num_heads, max_seq_length, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        # Load trained model weights (assuming you have a trained model)
        model.load_state_dict(torch.load('circuitformer_model.pth'))

        # Evaluate
        avg_loss, accuracy, all_predictions, all_targets = evaluate_model(model, test_loader, criterion, device)

        # Plot confusion matrix
        class_names = list(dataset.label_map.keys())
        plot_confusion_matrix(all_targets, all_predictions, class_names)

        print("Example evaluation completed.")
    else:
        print("Example evaluation code is available but not running.")
        print("Set the RUN_EXAMPLES environment variable to 'True' to run the example.")