"""Wide-window RNN model for financial stock price boost prediction.

Train and evaluate on live stock data (e.g. AAPL):

    python3 -m torchbox.finance --ticker AAPL --window-size 30 --epochs 10

Train using synthetic data:

    python3 -m torchbox.finance --synthetic --window-size 30 --epochs 10

Save and load trained model weights:

    python3 -m torchbox.finance --synthetic --save-model models/finance_rnn.pt
    python3 -m torchbox.finance --synthetic --load-model models/finance_rnn.pt --epochs 0

All commands also work via the torchbox CLI:

    torchbox finance --ticker MSFT --window-size 30 --epochs 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import requests
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def fetch_btc_data(
    use_direct_url: bool = True,
    **keys
) -> np.ndarray:
    """
        Keys: product_id="BTC-USD", n=100, granularity="ONE_HOUR"
    """
    if use_direct_url:
        # Query public spot price for BTC-USD
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            price = data["data"]["amount"]
            currency = data["data"]["currency"]
            print(f"The current Bitcoin price is {price} {currency}")
        else:
            print("Failed to fetch price:", response.status_code)
        return np.zeros(10)
    else:
        from datetime import datetime, timedelta, timezone
        from coinbase.rest import RESTClient

        # product_id="BTC-USD", n=100, granularity="ONE_HOUR"
        if "product_id" in keys:
            product_id = keys["product_id"]
        if "n" in keys:
            n = keys["n"]
        if "granularity" in keys:
            granularity = keys["granularity"]

        # Map granularity string to seconds to calculate bounds
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400
        }
        
        # Initialize the client (Public endpoints do not require active API keys)
        client = None
        config_path = Path.home() / ".config" / "apikeys.json"
        if config_path.exists():
            with config_path.open() as f:
                config_json = json.load(f)
                if "apikeys" in config_json \
                    and "coinbase" in config_json["apikeys"] \
                    and "mainkey01" in config_json["apikeys"]["coinbase"]:
                    client = RESTClient(api_key=config_json["apikeys"]["coinbase"]["mainkey01"]["name"],
                                        api_secret=config_json["apikeys"]["coinbase"]["mainkey01"]["key"])
                    print(f"Loaded API key from {str(config_path)}")
                else:
                    print("Could not find Coinbase Advanced API key in JSON structure")
                    return np.array([])
        else:
            client = RESTClient(
                api_key="organizations/24022f45-7d07-4580-82d5-c8f64c912957/apiKeys/9b9d7d4d-2fa0-4825-bc11-22ff7d0fe916",
                api_secret="jXqjL+5AxWnEfGi4fE8/QyHxwv9dtjfQZGk+vIZgr2JyVAGhhSOn4vHnB2iomlCnAVcjko5sGYPte/gL76s6aw==")
            print("Loading default API key")
    
        # Calculate timestamps
        end_time = datetime.now(timezone.utc)
        lookback = timedelta(seconds=granularity_seconds[granularity] * n)
        start_time = end_time - lookback

        # Convert datetime objects to string Unix timestamps as expected by the SDK
        start_str = str(int(start_time.timestamp()))
        end_str = str(int(end_time.timestamp()))

        # Fetch data using the SDK method
        response = client.get_candles(
            product_id=product_id,
            start=start_str,
            end=end_str,
            granularity=granularity
        )
        
        # Access the candles list from the returned object/dictionary
        candles = response["candles"]
        
        # Chronological sort (Coinbase returns them newest first)
        candles.reverse()
        
        return np.array([candle.open for candle in candles[:n]])

def fetch_stock_data(
    ticker: str = "AAPL",
    range_str: str = "1y",
    interval: str = "1d",
) -> np.ndarray:
    """Fetches historical stock closing prices from Yahoo Finance API.

    Args:
        ticker: Ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL').
        range_str: Historical range (e.g. '1y', '2y', '6m').
        interval: Data interval ('1d', '1wk').

    Returns:
        1D numpy array of historical closing prices.
    """
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    payload = response.json()
    results = payload.get("chart", {}).get("result", [])
    if not results:
        raise ValueError(f"No financial data found for ticker '{ticker}'.")

    quote = results[0]["indicators"]["quote"][0]
    closes = quote.get("close", [])
    valid_prices = [p for p in closes if p is not None and not np.isnan(p)]
    if not valid_prices:
        raise ValueError(f"No valid price entries found for ticker '{ticker}'.")

    return np.array(valid_prices, dtype=np.float32)


def generate_synthetic_stock_data(
    num_days: int = 500,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
    seed: int = 42,
) -> np.ndarray:
    """Generates synthetic stock price series via random walk with drift.

    Args:
        num_days: Number of daily price points to generate.
        initial_price: Starting price.
        volatility: Daily standard deviation of price changes.
        drift: Daily expected price return (drift).
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array of synthetic stock prices.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=volatility, size=num_days - 1)
    price_factors = np.cumprod(1.0 + returns)
    prices = np.empty(num_days, dtype=np.float32)
    prices[0] = initial_price
    prices[1:] = initial_price * price_factors
    return prices


class StockTimeSeriesDataset(Dataset):
    """Dataset for wide-window stock time series forecasting.

    Given a 1D sequence of prices and a window size W:
    - Input window X_i: prices [p_i, ..., p_{i+W-1}] (optionally normalized w.r.t. window start).
    - Target y_i: 1.0 if p_{i+W} > p_{i+W-1} (price boost in immediate next step), else 0.0.
    """

    def __init__(
        self,
        prices: np.ndarray,
        window_size: int = 30,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            prices: 1D array of stock prices.
            window_size: Number of time steps in the wide input window.
            normalize: If True, normalize input window by price returns relative to the first step.
        """
        self.window_size = window_size
        self.normalize = normalize

        prices = np.asarray(prices, dtype=np.float32)
        if len(prices) <= window_size:
            raise ValueError(
                f"Historical prices length ({len(prices)}) must be greater than window_size ({window_size})."
            )

        samples_list = []
        targets_list = []

        num_samples = len(prices) - window_size
        for i in range(num_samples):
            window = prices[i : i + window_size].copy()
            next_price = prices[i + window_size]
            last_window_price = window[-1]

            # Price boost condition: immediate next price > last price in the window
            target = 1.0 if next_price > last_window_price else 0.0

            if normalize:
                base_price = window[0] if window[0] != 0 else 1.0
                window = (window - base_price) / base_price

            samples_list.append(window[:, np.newaxis])  # shape: (window_size, 1)
            targets_list.append(target)

        self.samples = np.array(samples_list, dtype=np.float32)
        self.targets = np.array(targets_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_tensor = torch.from_numpy(self.samples[idx])
        target_tensor = torch.tensor([self.targets[idx]], dtype=torch.float32)
        return sample_tensor, target_tensor


class WideWindowRNN(nn.Module):
    """Wide-window Recurrent Neural Network for stock price boost prediction.

    Takes a wide window of historical stock time-series features (e.g. shape [batch_size, window_size, input_size])
    and outputs a single floating-point probability in [0, 1] representing the likelihood of a price boost
    in the immediate next time step.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        window_size: int = 30,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.rnn_type = rnn_type.lower()

        rnn_dict = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        rnn_cls = rnn_dict.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type '{rnn_type}'. Expected 'rnn', 'lstm', or 'gru'.")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, window_size, input_size) or (batch_size, window_size)

        Returns:
            Tensor of shape (batch_size, 1) with probabilities in range [0, 1].
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # shape: (batch_size, window_size, 1)

        out, _ = self.rnn(x)  # shape: (batch_size, window_size, hidden_size)
        last_step = out[:, -1, :]  # shape: (batch_size, hidden_size)
        hidden = self.relu(self.fc1(last_step))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return torch.sigmoid(logits)

    def predict_boost_probability(self, x: torch.Tensor | np.ndarray) -> float | np.ndarray:
        """Predict probability that there will be a stock price boost in the next step.

        Args:
            x: Sequence window tensor or numpy array of shape (window_size,), (window_size, input_size),
               or batch of shape (batch_size, window_size, input_size).

        Returns:
            A single float probability between 0 and 1 if input is a single sequence,
            or numpy array of float probabilities if input is a batch.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float()
            else:
                x_tensor = x.float()

            is_single = False
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0).unsqueeze(-1)  # (1, window_size, 1)
                is_single = True
            elif x_tensor.dim() == 2 and x_tensor.shape[0] == self.window_size:
                x_tensor = x_tensor.unsqueeze(0)  # (1, window_size, input_size)
                is_single = True

            probs = self.forward(x_tensor)
            if is_single or probs.shape[0] == 1:
                return float(probs.item())
            return probs.squeeze(-1).cpu().numpy()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains the model for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(data)
        total_samples += len(data)

    return running_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates model on validation/test set. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * len(data)
            predictions = (output >= 0.5).float()
            correct += int((predictions == target).sum().item())
            total_samples += len(data)

    avg_loss = running_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)
    return avg_loss, accuracy


def plot_financial_data(
    prices: np.ndarray,
    window_size: int = 30,
    output_path: str = "finance_samples.png",
    title: str = "Financial Training Data",
) -> None:
    """Generates and saves a plot of the financial training data using matplotlib.

    Args:
        prices: 1D numpy array of historical stock prices.
        window_size: Wide window size used by the model.
        output_path: Path to save the output plot image.
        title: Plot title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_len = min(len(prices), 200)
    subset_prices = prices[-display_len:]
    time_steps = np.arange(display_len)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot price curve
    ax.plot(time_steps, subset_prices, label="Stock Price", color="#1f77b4", linewidth=2.0)

    # Compute target boost indicators (immediate next price > current price)
    boosts = subset_prices[1:] > subset_prices[:-1]
    boost_indices = time_steps[:-1][boosts]
    no_boost_indices = time_steps[:-1][~boosts]

    ax.scatter(
        boost_indices,
        subset_prices[:-1][boosts],
        color="green",
        marker="^",
        s=35,
        label="Next Step Boost (Target = 1.0)",
        zorder=3,
    )
    ax.scatter(
        no_boost_indices,
        subset_prices[:-1][~boosts],
        color="red",
        marker="v",
        s=25,
        alpha=0.6,
        label="No Boost (Target = 0.0)",
        zorder=3,
    )

    # Highlight a sample wide window
    if display_len >= window_size:
        win_start = display_len - window_size - 1
        win_end = display_len - 1
        ax.axvspan(
            win_start,
            win_end,
            color="orange",
            alpha=0.2,
            label=f"Sample Wide Window (W={window_size})",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Steps (Days)", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    save_dir = os.path.dirname(output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved financial data sample plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for finance CLI."""
    parser = argparse.ArgumentParser(
        prog="torchbox finance",
        description="Train and evaluate a wide-window RNN stock price boost prediction model.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Wide window size in time steps (default: 30)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--rnn-type",
        type=str,
        choices=["rnn", "lstm", "gru"],
        default="lstm",
        help="RNN layer type: rnn, lstm, or gru (default: lstm)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic stock data instead of downloading online historical data",
    )
    parser.add_argument(
        "--dump-samples",
        nargs="?",
        const="finance_samples.png",
        default=None,
        metavar="FILE",
        help="Plot and save a sample of the financial training data using matplotlib (default: finance_samples.png)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model weights",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to load trained model weights",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA training",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="ONE_MINUTE",
        help="Set the training data granularity to extract"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for torchbox finance."""
    parser = build_parser()
    args = parser.parse_args(argv)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.synthetic:
        print("Generating synthetic stock price data...")
        prices = generate_synthetic_stock_data(num_days=500, initial_price=100.0)
    else:
        print(f"Fetching historical stock data for '{args.ticker}'...")
        try:
            #prices = fetch_stock_data(ticker=args.ticker)
            prices = fetch_btc_data(False, product_id="BTC-USD", n=350, granularity=args.granularity)
            print(f"Successfully fetched {len(prices)} price data points for {args.ticker}.")
        except (requests.RequestException, ValueError) as e:
            print(f"Failed to fetch stock data for {args.ticker} ({e}). Falling back to synthetic data.")
            prices = generate_synthetic_stock_data(num_days=500, initial_price=100.0)

    if args.dump_samples:
        title = f"Financial Training Data ({args.ticker if not args.synthetic else 'Synthetic'})"
        plot_financial_data(
            prices,
            window_size=args.window_size,
            output_path=args.dump_samples,
            title=title,
        )

    dataset = StockTimeSeriesDataset(prices, window_size=args.window_size, normalize=True)

    # Train / validation split (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = WideWindowRNN(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        window_size=args.window_size,
        rnn_type=args.rnn_type,
    ).to(device)

    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading weights from {args.load_model}...")
            model.load_state_dict(torch.load(args.load_model, weights_only=True, map_location=device))
        else:
            print(f"Warning: model file '{args.load_model}' not found.")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.epochs > 0:
        print(f"Training WideWindowRNN ({args.rnn_type.upper()}) for {args.epochs} epochs on {device}...")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:2d}/{args.epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
                )

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal Validation Loss: {val_loss:.4f} | Final Validation Accuracy: {val_acc * 100:.2f}%")

    # Run sample inference on the latest price window
    latest_window = dataset.samples[-1]  # shape: (window_size, 1)
    prob = model.predict_boost_probability(latest_window)
    print(f"\nImmediate next time step price boost probability: {prob:.4f}")

    if args.save_model:
        save_dir = os.path.dirname(args.save_model)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model weights to {args.save_model}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
