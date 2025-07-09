import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from classification.dino.feature_generator import (
    load_dinov2_model,
    preprocess_images_for_dinov2,
    compute_dinov2_feature_vectors,
)


# Define the MLP architecture
class ZDiscMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        # Layer dimensions based on checkpoint shapes
        # fc1: input_dim -> 512
        # fc2: 512 -> 256
        # fc22: 256 -> 256
        # fc3: 256 -> 128
        # fc4: 128 -> 64
        # fc5: 64 -> 2 (output)

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc22 = nn.Linear(256, 256)
        self.bn22 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 2)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x = self.drop(self.act(self.bn22(self.fc22(x))))
        x = self.drop(self.act(self.bn3(self.fc3(x))))
        x = self.drop(self.act(self.bn4(self.fc4(x))))
        return self.fc5(x)


def train_mlp_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 256,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-5,
):
    """
    Train a single MLP on provided train/test feature arrays.

    Args:
        train_features (np.ndarray): Shape (n_train, n_features)
        train_labels   (np.ndarray): Shape (n_train,), values 0 or 1
        test_features  (np.ndarray): Shape (n_test, n_features)
        test_labels    (np.ndarray): Shape (n_test,), values 0 or 1
        batch_size   (int): Batch size for DataLoader.
        num_epochs   (int): Number of training epochs.
        learning_rate  (float): Learning rate for Adam.
        dropout_rate (float): Dropout probability in each hidden layer.

    Returns:
        model (nn.Module): The trained MLP with best test accuracy loaded.
        best_accuracy (float): Highest test accuracy achieved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor(test_features, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    model = ZDiscMLP(input_dim=input_dim, dropout_rate=dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    best_test_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                test_loss += loss.item() * batch_x.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_state = model.state_dict()

        print(
            f"Epoch {epoch}/{num_epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Test Loss: {avg_test_loss:.4f}"
        )

    # Load best model and save checkpoint
    model.load_state_dict(best_state)
    print(f"\nBest Test Loss: {best_test_loss:.4f}")

    return model


def infer_zdisc_logits(images, mlp_checkpoint, batch_size=16):
    """
    Perform inference on a set of images and return the logits.

    Args:
        images (numpy.ndarray): Array of images (N, H, W, C).
        mlp_checkpoint (str): Path to the trained MLP checkpoint file.
        batch_size (int): Batch size for inference.
    Returns:
        numpy.ndarray: Logits for each image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # images is already a loaded numpy array
    print(f"Received {images.shape[0]} images with shape {images.shape[1:]}.")

    # Load the pretrained DINO model
    print("Loading DINO model...")
    dino_model = load_dinov2_model()

    # Transform images for DINO
    print("Transforming images for DINO...")
    transformed_images = preprocess_images_for_dinov2(images)

    # Extract feature vectors
    print("Extracting feature vectors with DINO...")
    feature_vectors = compute_dinov2_feature_vectors(
        dino_model, transformed_images, batch_size=batch_size
    )

    # Load the trained MLP model
    print("Loading trained MLP model...")
    input_dim = feature_vectors.shape[1]
    mlp_model = ZDiscMLP(input_dim=input_dim, dropout_rate=0.5)
    mlp_model.load_state_dict(torch.load(mlp_checkpoint, map_location=device))
    mlp_model.to(device)
    mlp_model.eval()

    # Perform inference
    print("Performing inference with MLP...")
    dataset = TensorDataset(torch.tensor(feature_vectors, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = mlp_model(inputs)
            all_logits.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    return all_logits
