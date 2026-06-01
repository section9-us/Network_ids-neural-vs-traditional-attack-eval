from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FlowMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [hidden_dim, max(hidden_dim // 2, 1)]
        layers: list[nn.Module] = []
        current_dim = input_dim
        for layer_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(current_dim, layer_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = layer_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class FlowAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dim: int = 64,
        classifier_hidden_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.autoencoder = FlowAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )
        self.classifier = FlowMLP(
            input_dim=latent_dim,
            hidden_layers=classifier_hidden_layers or [max(encoder_hidden_dim // 2, 1)],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encoder(x)
        return self.classifier(latent)


def neural_architecture_config(
    model_key: str,
    input_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    latent_dim: int = 16,
) -> dict:
    if model_key == "shallow_mlp":
        return {
            "model_type": "mlp",
            "display_name": "Shallow MLP",
            "hidden_layers": [hidden_dim],
            "dropout": dropout,
        }
    if model_key == "pytorch_mlp":
        return {
            "model_type": "mlp",
            "display_name": "PyTorch MLP",
            "hidden_layers": [hidden_dim, max(hidden_dim // 2, 1)],
            "dropout": dropout,
        }
    if model_key == "deep_mlp":
        return {
            "model_type": "mlp",
            "display_name": "Deep MLP",
            "hidden_layers": [
                hidden_dim * 2,
                hidden_dim,
                max(hidden_dim // 2, 1),
                max(hidden_dim // 4, 1),
            ],
            "dropout": dropout,
        }
    if model_key == "autoencoder_mlp":
        return {
            "model_type": "autoencoder_mlp",
            "display_name": "Autoencoder + MLP",
            "latent_dim": min(latent_dim, max(2, input_dim)),
            "encoder_hidden_dim": hidden_dim,
            "classifier_hidden_layers": [max(hidden_dim // 2, 1)],
            "dropout": dropout,
        }
    raise ValueError(f"Unknown neural model key: {model_key}")


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    hidden_layers: list[int] | None = None,
    dropout: float = 0.2,
    random_state: int = 42,
    display_name: str = "PyTorch MLP",
) -> FlowMLP:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FlowMLP(
        input_dim=x_train.shape[1],
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
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
            print(f"{display_name} epoch {epoch + 1:02d}/{epochs} - val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_autoencoder_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    autoencoder_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    latent_dim: int = 16,
    encoder_hidden_dim: int = 64,
    classifier_hidden_layers: list[int] | None = None,
    dropout: float = 0.2,
    random_state: int = 42,
) -> AutoencoderMLP:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderMLP(
        input_dim=x_train.shape[1],
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        classifier_hidden_layers=classifier_hidden_layers,
        dropout=dropout,
    ).to(device)

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    reconstruction_loss = nn.MSELoss()
    ae_optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=learning_rate)

    for epoch in range(autoencoder_epochs):
        model.autoencoder.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            ae_optimizer.zero_grad()
            reconstructed = model.autoencoder(xb)
            loss = reconstruction_loss(reconstructed, xb)
            loss.backward()
            ae_optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Autoencoder pretrain epoch {epoch + 1:02d}/{autoencoder_epochs}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    classifier_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    classifier_loader = DataLoader(classifier_ds, batch_size=batch_size, shuffle=True)
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for xb, yb in classifier_loader:
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
            print(f"Autoencoder + MLP epoch {epoch + 1:02d}/{epochs} - val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_neural_model(
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    autoencoder_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    dropout: float = 0.2,
    random_state: int = 42,
) -> tuple[nn.Module, dict]:
    config = neural_architecture_config(
        model_key,
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        latent_dim=latent_dim,
    )
    if config["model_type"] == "mlp":
        model = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_layers=config["hidden_layers"],
            dropout=dropout,
            random_state=random_state,
            display_name=config["display_name"],
        )
    elif config["model_type"] == "autoencoder_mlp":
        model = train_autoencoder_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            autoencoder_epochs=autoencoder_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            latent_dim=config["latent_dim"],
            encoder_hidden_dim=config["encoder_hidden_dim"],
            classifier_hidden_layers=config["classifier_hidden_layers"],
            dropout=dropout,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported neural model type: {config['model_type']}")
    return model, config


def evaluate_loss(model: nn.Module, x_val, y_val, loss_fn, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_val, dtype=torch.float32, device=device)
        y = torch.tensor(y_val, dtype=torch.float32, device=device)
        logits = model(x)
        return float(loss_fn(logits, y).item())


def predict_mlp(model: nn.Module, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probabilities = predict_mlp_proba(model, x)
    return (probabilities >= threshold).astype(int)


def predict_mlp_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(model(tensor_x)).cpu().numpy()
    return probabilities


def save_mlp_checkpoint(
    model: nn.Module,
    path: Path,
    input_dim: int,
    config: dict | None = None,
    threshold: float = 0.5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = config or {
        "model_type": "mlp",
        "display_name": "PyTorch MLP",
        "hidden_layers": [64, 32],
        "dropout": 0.2,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "config": config,
            "threshold": threshold,
        },
        path,
    )


def load_mlp_checkpoint(path: Path) -> tuple[nn.Module, float]:
    checkpoint = torch.load(path, map_location="cpu")
    input_dim = int(checkpoint["input_dim"])
    config = checkpoint.get("config")
    if config is None:
        config = {
            "model_type": "mlp",
            "display_name": "PyTorch MLP",
            "hidden_layers": [
                int(checkpoint.get("hidden_dim", 64)),
                max(int(checkpoint.get("hidden_dim", 64)) // 2, 1),
            ],
            "dropout": float(checkpoint.get("dropout", 0.2)),
        }

    if config["model_type"] == "mlp":
        model = FlowMLP(
            input_dim=input_dim,
            hidden_layers=list(config["hidden_layers"]),
            dropout=float(config.get("dropout", 0.2)),
        )
    elif config["model_type"] == "autoencoder_mlp":
        model = AutoencoderMLP(
            input_dim=input_dim,
            latent_dim=int(config["latent_dim"]),
            encoder_hidden_dim=int(config["encoder_hidden_dim"]),
            classifier_hidden_layers=list(config["classifier_hidden_layers"]),
            dropout=float(config.get("dropout", 0.2)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {config['model_type']}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint.get("threshold", 0.5))
