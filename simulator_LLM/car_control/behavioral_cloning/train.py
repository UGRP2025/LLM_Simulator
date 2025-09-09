import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

from model import create_model
from dataset import DrivingDataset, ControlBins

# --- Parameters ---
DATA_PATH = os.path.expanduser('~/f1tenth_data')
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

def train():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path does not exist: {DATA_PATH}")
        print("Please run data_collector.py to generate data first.")
        return

    control_bins = ControlBins()
    dataset = DrivingDataset(data_path=DATA_PATH, transform=transform, control_bins=control_bins)
    
    if len(dataset) == 0:
        print("Dataset is empty. Please collect data before training.")
        return

    # Split dataset into training and validation
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")

    # Model, Loss, Optimizer
    num_classes = control_bins.num_bins
    model = create_model(num_outputs=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # Only train the new layer

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # --- Save Model ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_save_path = os.path.join(script_dir, "behavioral_cloning_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Finished Training. Model saved to {model_save_path}")

if __name__ == '__main__':
    train()
