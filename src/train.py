import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

def get_lr_scheduler(optimizer, warmup_steps=4000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps ** 0.5) * (step ** -0.5)
    return LambdaLR(optimizer, lr_lambda)

def train_circuitformer(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scheduler = get_lr_scheduler(optimizer)
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)  # Ignore attention weights during training
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)  # Ignore attention weights during validation
                val_loss += criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return train_losses, val_losses, accuracies