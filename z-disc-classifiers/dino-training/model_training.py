import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc5 = nn.Linear(64, 2)  # Binary classification
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout1(self.activation(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout2(self.activation(self.fc2(x)))
        x = self.bn2(x)
        x = self.dropout3(self.activation(self.fc3(x)))
        x = self.bn3(x)
        x = self.dropout4(self.activation(self.fc4(x)))
        x = self.bn4(x)
        x = self.fc5(x)
        return x


# Training function
def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on test data
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_accuracy = correct / total
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = model.state_dict()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}")

    print(f"Best Test Accuracy: {best_test_accuracy:.4f}")
    model.load_state_dict(best_model_state)
    return model, best_test_accuracy


# Main function to train and save models
def main(inputs, labels, output_dir="models", num_splits=5):
    os.makedirs(output_dir, exist_ok=True)

    input_dim = inputs.shape[1]
    all_accuracies = []

    for i in range(num_splits):
        print(f"\nGenerating split {i + 1}/{num_splits}...")
        # Set random seed for reproducibility
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=0.2, random_state=seed, stratify=labels
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False)

        # Initialize and train the model
        model = MLP(input_dim=input_dim, dropout_rate=0.5)
        model, best_accuracy = train_model(model, train_loader, test_loader)
        all_accuracies.append(best_accuracy)

        # Save the best model
        model_path = os.path.join(output_dir, f"model_{i + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model {i + 1} saved to {model_path}, Best Accuracy: {best_accuracy:.4f}")

    # Summary of all accuracies
    print("\nTraining complete.")
    for i, accuracy in enumerate(all_accuracies):
        print(f"Model {i + 1}: Best Test Accuracy = {accuracy:.4f}")


# Example usage
# Replace these with your actual data
inputs = ...  # Example: Feature vectors (N x D)
labels = ...  # Example: Binary labels (N,)
main(inputs, labels)
