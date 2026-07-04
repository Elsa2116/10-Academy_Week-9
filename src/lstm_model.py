"""
Deep-learning forecasting: LSTM model for Tesla (TSLA) closing-price
sequence prediction. Implemented in PyTorch (CPU-only).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(42)
torch.set_num_threads(1)


class LSTMForecaster(nn.Module):
    """2-layer stacked LSTM + dropout + dense head, mirroring a typical Keras LSTM stack."""

    def __init__(self, window: int = 60, hidden_size: int = 50, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def make_sequences(scaled_series: np.ndarray, window: int = 60):
    """
    Convert a 1-D scaled series into (X, y) supervised-learning sequences,
    where X[i] is `window` consecutive points and y[i] is the point
    immediately after them.
    """
    X, y = [], []
    for i in range(window, len(scaled_series)):
        X.append(scaled_series[i - window:i, 0])
        y.append(scaled_series[i, 0])
    X, y = np.array(X, dtype="float32"), np.array(y, dtype="float32")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm_model(window: int = 60, units: int = 50, dropout: float = 0.2, learning_rate: float = 0.001):
    """Build an LSTMForecaster instance (PyTorch nn.Module)."""
    return LSTMForecaster(window=window, hidden_size=units, dropout=dropout)


def train_test_scale(train_prices: np.ndarray, test_prices: np.ndarray):
    """Fit a MinMaxScaler on training prices and transform both splits."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices.reshape(-1, 1))
    test_scaled = scaler.transform(test_prices.reshape(-1, 1))
    return train_scaled, test_scaled, scaler


def fit(model, X_train, y_train, epochs=15, batch_size=32, learning_rate=0.001, validation_split=0.1):
    """
    Train the PyTorch LSTM model with mini-batch gradient descent.
    Returns a history dict with 'loss' and 'val_loss' lists (mirrors Keras API).
    """
    n = len(X_train)
    n_val = int(n * validation_split)
    n_train = n - n_val

    X_t = torch.from_numpy(X_train[:n_train])
    y_t = torch.from_numpy(y_train[:n_train]).unsqueeze(1)
    X_v = torch.from_numpy(X_train[n_train:])
    y_v = torch.from_numpy(y_train[n_train:]).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = {"loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v) if n_val > 0 else None
            val_loss = float(loss_fn(val_pred, y_v).item()) if n_val > 0 else train_loss

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return history


def predict(model, X):
    """Run inference on a batch of sequences; returns a numpy array of predictions."""
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype("float32"))
        preds = model(X_t).numpy()
    return preds


def iterative_forecast(model, last_window: np.ndarray, steps: int, scaler):
    """
    Iteratively predict `steps` future points, feeding each prediction back
    in as input for the next step (multi-step forecasting).
    `last_window` should be scaled and shaped (window, 1).
    """
    window = last_window.copy()
    preds_scaled = []
    model.eval()
    for _ in range(steps):
        x = torch.from_numpy(window.reshape(1, window.shape[0], 1).astype("float32"))
        with torch.no_grad():
            next_scaled = model(x).numpy()[0, 0]
        preds_scaled.append(next_scaled)
        window = np.append(window[1:], [[next_scaled]], axis=0)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
