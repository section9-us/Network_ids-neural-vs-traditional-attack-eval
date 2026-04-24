from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FlowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> FlowMLP:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowMLP(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, x_val, y_val, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"PyTorch epoch {epoch + 1:02d}/{epochs} - val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_loss(model: FlowMLP, x_val, y_val, loss_fn, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_val, dtype=torch.float32, device=device)
        y = torch.tensor(y_val, dtype=torch.float32, device=device)
        logits = model(x)
        return float(loss_fn(logits, y).item())


def predict_mlp(model: FlowMLP, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(model(tensor_x)).cpu().numpy()
    return (probabilities >= threshold).astype(int)
