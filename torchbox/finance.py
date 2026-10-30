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
import json
import tqdm
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from google.colab import userdata

import numpy as np
import requests
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

try:
    from coinbase.rest import RESTClient
except ImportError:
    RESTClient = None

MAX_REQUEST_SAMPLES = 350
DEFAULT_CACHE_FILE = "data/btc_cache.npy"


def load_cached_btc_data(cache_path: str | Path) -> np.ndarray:
    """Loads cached BTC time series data from a local file (.npy, .csv, or .json)."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return np.asarray(data, dtype=np.float32)
    elif path.suffix in (".csv", ".txt"):
        return np.loadtxt(path, delimiter=",", dtype=np.float32)
    else:
        return np.asarray(np.load(path), dtype=np.float32)


def save_cached_btc_data(cache_path: str | Path, data: np.ndarray) -> None:
    """Saves BTC time series data to a local cache file (.npy, .csv, or .json)."""
    path = Path(cache_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(arr.tolist(), f)
    elif path.suffix in (".csv", ".txt"):
        np.savetxt(path, arr, delimiter=",")
    else:
        np.save(path, arr)


def fetch_btc_data(
    use_direct_url: bool = True,
    cache_file: str | Path | None = DEFAULT_CACHE_FILE,
    **keys,
) -> np.ndarray:
    """Fetches BTC time series data, checking local cache file first.

    Keys: product_id="BTC-USD", n=100, granularity="ONE_HOUR", cache_file=...
    """
    if "cache_file" in keys:
        cache_file = keys["cache_file"]

    # If a local cache file exists, load it directly without running fetch.
    if cache_file is not None:
        cache_path = Path(cache_file)
        if not cache_path.exists() and cache_file == DEFAULT_CACHE_FILE:
            alt_path = Path("btc_cache.npy")
            if alt_path.exists():
                cache_path = alt_path
        if cache_path.exists():
            print(f"Loading cached BTC data from {cache_path}...")
            return load_cached_btc_data(cache_path)

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
        # product_id="BTC-USD", n=100, granularity="ONE_HOUR"
        product_id = keys.get("product_id", "BTC-USD")
        n = keys.get("n", 100)
        granularity = keys.get("granularity", "ONE_HOUR")

        # Map granularity string to seconds to calculate bounds
        granularity_seconds = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }

        if RESTClient is None:
            raise ImportError(
                "coinbase-advanced-py is not installed. Install it via 'pip install coinbase-advanced-py' "
                "or provide a local cache file."
            )

        # Initialize the client (Public endpoints do not require active API keys)
        client = None
        config_path = Path.home() / ".config" / "apikeys.json"
        if userdata is not None and "coinbase_api_key" in userdata and "coinbase_api_secret" in userdata:
            client = RESTClient(
                api_key=userdata.get("coinbase_api_key"),
                api_secret=userdata.get("coinbase_api_secret")
            )
            print("Loaded API key from colab secrets")
        elif config_path.exists():
            with config_path.open() as f:
                config_json = json.load(f)
                if (
                    "apikeys" in config_json
                    and "coinbase" in config_json["apikeys"]
                    and "mainkey01" in config_json["apikeys"]["coinbase"]
                ):
                    client = RESTClient(
                        api_key=config_json["apikeys"]["coinbase"]["mainkey01"]["name"],
                        api_secret=config_json["apikeys"]["coinbase"]["mainkey01"]["key"],
                    )
                    print(f"Loaded API key from {config_path!s}")
                else:
                    print("Could not find Coinbase Advanced API key in JSON structure")
                    return np.array([])
        else:
            client = RESTClient(
                api_key="organizations/24022f45-7d07-4580-82d5-c8f64c912957/apiKeys/9b9d7d4d-2fa0-4825-bc11-22ff7d0fe916",
                api_secret="jXqjL+5AxWnEfGi4fE8/QyHxwv9dtjfQZGk+vIZgr2JyVAGhhSOn4vHnB2iomlCnAVcjko5sGYPte/gL76s6aw==",
            )
            print("Loading default API key")

        # Calculate timestamps
        end_time = datetime.now(timezone.utc)
        lookback = timedelta(seconds=granularity_seconds[granularity] * n)
        start_time = end_time - lookback

        # Fetch data using the SDK method
        candles = []
        candles_times = []
        start_time_int = int(start_time.timestamp())
        end_time_int = int(end_time.timestamp())
        for start, end in tqdm.tqdm([
            (i, min(i + MAX_REQUEST_SAMPLES, end_time_int))
            for i in range(int(start_time_int), end_time_int, MAX_REQUEST_SAMPLES)
        ]):
            start_str = str(start)
            end_str = str(end)
            response = client.get_candles(
                product_id=product_id,
                start=start_str,
                end=end_str,
                granularity=granularity,
            )

            # Access the candles list from the returned object/dictionary
            these_candles = response["candles"]
            these_candles.reverse()
            candles = candles + [candle.open for candle in these_candles]
            candles_times = candles_times + [candle.start for candle in these_candles]

        if not all((candles_times[i] < candles_times[i + 1]) for i in range(len(candles_times) - 1)):
            raise ValueError("Sample timestamps are not consecutive")

        prices = np.array(candles, dtype=np.float32)
        if cache_file is not None and len(prices) > 0:
            cache_path = Path(cache_file)
            save_cached_btc_data(cache_path, prices)
            print(f"Cached {len(prices)} BTC data points to {cache_path}.")

        return prices

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


def build_window_features(
    window: np.ndarray,
    scales: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Builds a multi-channel feature matrix for a single raw price window.

    Channel 0 is always the window normalized as a return relative to its first
    price (leak-free: uses only in-window information). When ``scales`` is
    provided, one additional channel per scale ``s`` is appended, holding the
    ``s``-step trailing return at each position -- this gives the model
    multi-time-scale context at a constant (short) window length instead of
    requiring a very long window.

    Args:
        window: 1D array of raw prices, shape (window_size,).
        scales: Optional tuple of look-back horizons (in steps) for the extra
            return channels. Scales >= window length are skipped.

    Returns:
        Float32 array of shape (window_size, 1 + num_valid_scales).
    """
    window = np.asarray(window, dtype=np.float32)
    base = window[0] if window[0] != 0 else 1.0
    channels = [(window - base) / base]

    for s in scales or ():
        if s < 1 or s >= len(window):
            continue
        ret = np.zeros_like(window)
        denom = window[:-s]
        denom = np.where(denom == 0, 1.0, denom)
        ret[s:] = (window[s:] - window[:-s]) / denom
        channels.append(ret)

    return np.stack(channels, axis=-1).astype(np.float32)  # (W, num_channels)


class StockTimeSeriesDataset(Dataset):
    """Dataset for wide-window stock time series forecasting.

    Given a 1D sequence of prices, a window size W, a smoothing horizon P and a
    dead-band epsilon:
    - Input window X_i: prices [p_i, ..., p_{i+W-1}], encoded as one or more
      feature channels by :func:`build_window_features`.
    - Future value f_i: mean (or EMA) of the next P prices
      [p_{i+W}, ..., p_{i+W+P-1}]. Smoothing over P cancels tick noise so the
      target reflects genuine short-horizon drift rather than a single noisy tick.
    - Target y_i: 1.0 if the smoothed future exceeds the last window price by
      more than the dead-band, else 0.0. Samples whose relative move falls inside
      the dead-band are dropped as noise (not used for training/eval).
    """

    def __init__(
        self,
        prices: np.ndarray,
        window_size: int = 30,
        normalize: bool = True,
        smooth_p: int = 1,
        dead_band: float = 0.0,
        smooth_mode: str = "mean",
        scales: tuple[int, ...] | None = None,
    ) -> None:
        """
        Args:
            prices: 1D array of stock prices.
            window_size: Number of time steps in the wide input window.
            normalize: If True, normalize input window by returns relative to the
                first step. (Always on when scales are used; kept for API parity.)
            smooth_p: Number of future steps P to smooth over when forming the
                target. P=1 reproduces the original immediate-next-step target.
            dead_band: Minimum absolute relative move (e.g. 0.001 = 0.1%) for a
                sample to count. Smaller moves are discarded as noise.
            smooth_mode: 'mean' for a simple average of the next P prices, or
                'ema' for an exponentially weighted average (recent steps weighted
                more heavily).
            scales: Optional look-back horizons for extra return channels. When
                None, a single normalized-price channel is used (input_size=1).
        """
        if smooth_p < 1:
            raise ValueError(f"smooth_p must be >= 1, got {smooth_p}.")
        if dead_band < 0:
            raise ValueError(f"dead_band must be >= 0, got {dead_band}.")
        if smooth_mode not in ("mean", "ema"):
            raise ValueError(f"smooth_mode must be 'mean' or 'ema', got '{smooth_mode}'.")

        self.window_size = window_size
        self.normalize = normalize
        self.smooth_p = smooth_p
        self.dead_band = dead_band
        self.smooth_mode = smooth_mode
        self.scales = tuple(scales) if scales else ()

        prices = np.asarray(prices, dtype=np.float32)
        if len(prices) < window_size + smooth_p:
            raise ValueError(
                f"Historical prices length ({len(prices)}) must be at least "
                f"window_size + smooth_p ({window_size} + {smooth_p})."
            )

        # EMA weights over the P future steps (most recent step weighted highest).
        if smooth_mode == "ema" and smooth_p > 1:
            alpha = 2.0 / (smooth_p + 1.0)
            offsets = np.arange(smooth_p, dtype=np.float32)  # 0 == nearest future step
            ema_weights = (1.0 - alpha) ** offsets
            ema_weights = ema_weights / ema_weights.sum()
        else:
            ema_weights = None

        samples_list = []
        targets_list = []
        returns_list = []

        num_samples = len(prices) - window_size - smooth_p + 1
        for i in range(num_samples):
            window = prices[i : i + window_size].copy()
            future = prices[i + window_size : i + window_size + smooth_p]
            last_window_price = window[-1]

            if ema_weights is not None:
                future_val = float(np.dot(future, ema_weights))
            else:
                future_val = float(future.mean())

            base = last_window_price if last_window_price != 0 else 1.0
            rel_move = (future_val - last_window_price) / base

            # Dead-band: discard sub-threshold moves as noise rather than training on them.
            if abs(rel_move) < dead_band:
                continue

            target = 1.0 if rel_move > 0 else 0.0

            if normalize or self.scales:
                features = build_window_features(window, self.scales)
            else:
                features = window[:, np.newaxis]

            samples_list.append(features)
            targets_list.append(target)
            # Realized relative move over the smoothed horizon -- the P&L a trade
            # entered at the last window price would capture (before costs).
            returns_list.append(rel_move)

        if not samples_list:
            raise ValueError(
                "No samples survived the dead-band filter; lower --dead-band or "
                "provide more data."
            )

        self.samples = np.array(samples_list, dtype=np.float32)
        self.targets = np.array(targets_list, dtype=np.float32)
        self.returns = np.array(returns_list, dtype=np.float32)
        self.input_size = self.samples.shape[-1]

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


def _roc_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    """Computes ROC AUC via the Mann-Whitney rank statistic (ties averaged).

    Dependency-free alternative to sklearn.metrics.roc_auc_score. Returns NaN if
    either class is absent (AUC undefined).
    """
    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty(len(probs), dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)

    # Average ranks within tied probability groups.
    sorted_probs = probs[order]
    i = 0
    while i < len(sorted_probs):
        j = i
        while j + 1 < len(sorted_probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    sum_pos_ranks = ranks[targets == 1].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates the model and returns a dict of metrics.

    Keys: loss, accuracy, baseline (majority-class accuracy), precision, recall,
    auc, pos_rate. Accuracy is only meaningful relative to ``baseline`` -- a model
    that always predicts the majority class scores ``baseline`` while learning
    nothing, which is the usual trap in price-direction prediction.
    """
    model.eval()
    running_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * len(data)
            all_probs.append(output.squeeze(-1).cpu().numpy())
            all_targets.append(target.squeeze(-1).cpu().numpy())

    if not all_targets:
        return {
            "loss": 0.0, "accuracy": 0.0, "baseline": 0.0,
            "precision": 0.0, "recall": 0.0, "auc": float("nan"), "pos_rate": 0.0,
        }

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs >= 0.5).astype(np.float32)
    total = len(targets)

    accuracy = float((preds == targets).mean())
    pos_rate = float(targets.mean())
    # Majority-class baseline: accuracy of always predicting the more common label.
    baseline = max(pos_rate, 1.0 - pos_rate)

    tp = float(((preds == 1) & (targets == 1)).sum())
    fp = float(((preds == 1) & (targets == 0)).sum())
    fn = float(((preds == 0) & (targets == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": accuracy,
        "baseline": baseline,
        "precision": precision,
        "recall": recall,
        "auc": _roc_auc(probs, targets),
        "pos_rate": pos_rate,
    }


def collect_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Returns the model's output probabilities for a loader, in iteration order.

    Use with a ``shuffle=False`` loader so the probabilities line up positionally
    with the dataset indices behind it.
    """
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            chunks.append(model(data).squeeze(-1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def profitability_report(
    probs: np.ndarray,
    returns: np.ndarray,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    threshold: float = 0.5,
    allow_short: bool = False,
) -> dict[str, float]:
    """Cost-aware P&L sanity check for a signal over its realized returns.

    Each sample is treated as an independent round-trip trade: enter when the
    model's probability clears ``threshold`` (and, with ``allow_short``, enter the
    opposite side otherwise), capture that sample's realized relative move, and pay
    a round-trip cost of ``2 * (fee_bps + slippage_bps)``. This is a signal-level
    sanity check, not a full mark-to-market backtest -- the stride-1 samples
    overlap in time, so compounded figures are indicative, not executable.

    The ``always-long`` benchmark applies the same accounting to a position that is
    long every sample, so the comparison is apples-to-apples: the model only earns
    its keep if selective trading beats simply being long after costs.

    Args:
        probs: Model probabilities, aligned with ``returns``.
        returns: Realized relative moves per sample (e.g. dataset.returns[idx]).
        fee_bps: Taker fee per side, in basis points (1 bp = 0.01%).
        slippage_bps: Assumed slippage per side, in basis points.
        threshold: Probability cutoff for taking a long position.
        allow_short: If True, go short (-1) below threshold instead of flat (0).

    Returns:
        Dict of P&L metrics (fractional returns; *_bps fields in basis points).
    """
    probs = np.asarray(probs, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 1e4

    long_mask = probs >= threshold
    signal = np.where(long_mask, 1.0, -1.0 if allow_short else 0.0)
    traded = signal != 0

    gross = signal * returns
    cost = np.abs(signal) * round_trip_cost
    net = gross - cost
    net_traded = net[traded]
    n_trades = int(traded.sum())

    strat_net = float(net.sum())
    strat_equity = float(np.prod(1.0 + net_traded) - 1.0) if n_trades else 0.0
    avg_trade_bps = float(net_traded.mean() * 1e4) if n_trades else 0.0
    win_rate = float((net_traded > 0).mean()) if n_trades else 0.0
    if n_trades > 1 and net_traded.std() > 0:
        sharpe = float(net_traded.mean() / net_traded.std())
    else:
        sharpe = float("nan")

    # Always-long benchmark under identical cost accounting.
    bh_net = float((returns - round_trip_cost).sum())

    return {
        "net_return": strat_net,
        "net_equity": strat_equity,
        "avg_trade_bps": avg_trade_bps,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "n_trades": float(n_trades),
        "n_samples": float(len(returns)),
        "round_trip_bps": round_trip_cost * 1e4,
        "benchmark_net": bh_net,
        "excess_net": strat_net - bh_net,
    }


def walk_forward_folds(
    n_samples: int,
    n_folds: int,
    scheme: str = "expanding",
    embargo: int = 0,
    train_size: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Builds chronological train/test index folds for rolling-origin evaluation.

    The sample timeline is divided into ``n_folds + 1`` equal blocks; the first
    block seeds the initial training set and each subsequent block becomes one
    test fold, always strictly later in time than its training data.

    An ``embargo`` gap of samples is purged from the end of each training set so
    that overlapping windows (which share prices with the test block's inputs or
    target horizon) cannot leak across the boundary.

    Args:
        n_samples: Total number of chronologically-ordered samples.
        n_folds: Number of test folds to produce.
        scheme: 'expanding' grows the training set each fold; 'sliding' keeps a
            fixed-size trailing training window (requires ``train_size``).
        embargo: Samples to drop from the tail of each training set.
        train_size: Fixed training length for the 'sliding' scheme.

    Returns:
        List of (train_indices, test_indices) tuples; folds too small to use are
        skipped.
    """
    block = n_samples // (n_folds + 1)
    if block < 1:
        return []

    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_folds):
        test_start = (k + 1) * block
        test_end = n_samples if k == n_folds - 1 else test_start + block
        train_end = max(0, test_start - embargo)

        if scheme == "sliding" and train_size:
            train_start = max(0, train_end - train_size)
        else:
            train_start = 0

        if train_end - train_start < 1 or test_end - test_start < 1:
            continue
        folds.append((list(range(train_start, train_end)), list(range(test_start, test_end))))

    return folds


def run_fold(
    dataset: Dataset,
    train_idx: list[int],
    val_idx: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    """Trains a fresh model on one fold and returns its validation metrics."""
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False
    )

    model = WideWindowRNN(
        input_size=dataset.input_size,
        hidden_size=64,
        num_layers=3,
        window_size=args.window_size,
        rnn_type=args.rnn_type,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for _ in range(max(args.epochs, 1)):
        train_epoch(model, train_loader, optimizer, criterion, device)

    metrics = evaluate(model, val_loader, criterion, device)

    # Cost-aware P&L on this fold's test block (val_loader is unshuffled, so the
    # probabilities line up with dataset.returns[val_idx]).
    probs = collect_probs(model, val_loader, device)
    fold_returns = dataset.returns[val_idx]
    metrics.update(
        profitability_report(
            probs,
            fold_returns,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            threshold=args.threshold,
            allow_short=args.allow_short,
        )
    )
    return metrics


def walk_forward_evaluate(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float] | None:
    """Runs rolling-origin cross-validation and prints per-fold + aggregate metrics.

    Returns a dict of aggregate (mean) metrics, or None if no usable folds could
    be constructed.
    """
    embargo = args.embargo if args.embargo >= 0 else (args.window_size + args.smooth_p - 1)
    folds = walk_forward_folds(
        len(dataset),
        n_folds=args.walk_forward,
        scheme=args.wf_scheme,
        embargo=embargo,
        train_size=args.wf_train_size,
    )
    if not folds:
        print(
            f"Walk-forward: not enough samples ({len(dataset)}) for {args.walk_forward} "
            f"folds with embargo {embargo}; skipping."
        )
        return None

    print(
        f"\nRolling-origin evaluation: {len(folds)} folds | scheme={args.wf_scheme} | "
        f"embargo={embargo} samples"
    )
    print(
        f"  {'fold':>4} | {'train':>6} | {'test':>5} | {'acc':>6} | {'base':>6} | "
        f"{'edge':>6} | {'auc':>5}"
    )

    per_fold: list[dict[str, float]] = []
    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        m = run_fold(dataset, train_idx, val_idx, args, device)
        m["edge"] = m["accuracy"] - m["baseline"]
        per_fold.append(m)
        print(
            f"  {i:>4} | {len(train_idx):>6} | {len(val_idx):>5} | "
            f"{m['accuracy'] * 100:>5.1f}% | {m['baseline'] * 100:>5.1f}% | "
            f"{m['edge'] * 100:>+5.1f} | {m['auc']:>5.3f}"
        )

    def _agg(key: str) -> tuple[float, float]:
        vals = np.array([f[key] for f in per_fold], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std())

    acc_m, acc_s = _agg("accuracy")
    edge_m, edge_s = _agg("edge")
    auc_m, auc_s = _agg("auc")
    print(
        f"\n  Aggregate across {len(per_fold)} folds (mean +/- std):\n"
        f"    Accuracy    : {acc_m * 100:.2f}% +/- {acc_s * 100:.2f}\n"
        f"    Edge vs base: {edge_m * 100:+.2f} +/- {edge_s * 100:.2f} pts  "
        f"<-- consistently > 0 across folds is what you want\n"
        f"    ROC AUC     : {auc_m:.3f} +/- {auc_s:.3f}  (0.5 = no skill)"
    )
    return {"accuracy": acc_m, "edge": edge_m, "auc": auc_m}


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

def parse_tuple(value):
    try:
        # Split the string by commas and convert elements to integers
        return tuple(map(int, value.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuple must be comma-separated integers (e.g., '1,2,3')")

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
    parser.add_argument(
        "--smooth-p",
        type=int,
        default=1,
        help="Smooth the target over the next P future steps (default: 1 = immediate next step)",
    )
    parser.add_argument(
        "--smooth-mode",
        type=str,
        choices=["mean", "ema"],
        default="mean",
        help="How to smooth the next P steps for the target: 'mean' or 'ema' (default: mean)",
    )
    parser.add_argument(
        "--dead-band",
        type=float,
        default=0.0,
        help="Discard samples whose smoothed relative move is below this threshold "
             "(e.g. 0.001 = 0.1%%) as noise (default: 0.0 = keep all)",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Add multi-time-scale trailing-return channels to each window "
             "(see --scales) instead of a single normalized-price channel",
    )
    parser.add_argument(
        "--scales",
        type=parse_tuple,
        default=(1, 5, 15),
        help="Look-back horizons (in steps) for --multi-scale return channels (default: 1 5 15)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=700,
        help="Number of samples to load from stock/crypto API"
    )
    parser.add_argument(
        "--walk-forward",
        type=int,
        default=0,
        metavar="N",
        help="Run rolling-origin (walk-forward) evaluation with N folds instead of trusting "
             "a single split (default: 0 = off)",
    )
    parser.add_argument(
        "--wf-scheme",
        type=str,
        choices=["expanding", "sliding"],
        default="expanding",
        help="Walk-forward training scheme: 'expanding' (growing history) or 'sliding' "
             "(fixed trailing window) (default: expanding)",
    )
    parser.add_argument(
        "--wf-train-size",
        type=int,
        default=None,
        help="Fixed training length (in samples) for --wf-scheme sliding",
    )
    parser.add_argument(
        "--embargo",
        type=int,
        default=-1,
        help="Samples to purge between each walk-forward train/test boundary to prevent "
             "overlapping-window leakage (default: -1 = auto = window_size + smooth_p - 1)",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=10.0,
        help="Taker fee per side in basis points for the profitability check (default: 10 = 0.10%%)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Assumed slippage per side in basis points for the profitability check (default: 5 = 0.05%%)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for taking a long position in the profitability check (default: 0.5)",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="In the profitability check, go short below the threshold instead of flat",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=DEFAULT_CACHE_FILE,
        help=f"Path to local cache file for BTC time series data (default: {DEFAULT_CACHE_FILE})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading from or saving to local cache file",
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
            cache_file = None if args.no_cache else args.cache_file
            prices = fetch_btc_data(
                False,
                product_id=args.ticker,
                n=args.samples,
                granularity=args.granularity,
                cache_file=cache_file,
            )
            print(f"Successfully loaded {len(prices)} price data points for {args.ticker}.")
        except (requests.RequestException, ValueError, OSError, ImportError) as e:
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

    scales = args.scales if args.multi_scale else None
    dataset = StockTimeSeriesDataset(
        prices,
        window_size=args.window_size,
        normalize=True,
        smooth_p=args.smooth_p,
        dead_band=args.dead_band,
        smooth_mode=args.smooth_mode,
        scales=scales,
    )
    print(
        f"Built {len(dataset)} samples | input_size={dataset.input_size} | "
        f"smooth_p={args.smooth_p} ({args.smooth_mode}) | dead_band={args.dead_band}"
    )

    # Rolling-origin evaluation mode: trains a fresh model per fold across the
    # timeline, so there is no single model to save/infer -- report and return.
    if args.walk_forward > 0:
        if args.epochs <= 0:
            print("Walk-forward requires --epochs > 0 to train each fold; aborting.")
            return 1
        walk_forward_evaluate(dataset, args, device)
        return 0

    # Chronological train / validation split (first 80% train, last 20% val).
    # A time-ordered split avoids leaking future windows into training, which a
    # shuffled split would do and which silently inflates validation metrics.
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = WideWindowRNN(
        input_size=dataset.input_size,
        hidden_size=64,
        num_layers=3,
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
            metrics = evaluate(model, val_loader, criterion, device)
            if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:2d}/{args.epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
                    f"Val Acc: {metrics['accuracy'] * 100:.2f}% (baseline {metrics['baseline'] * 100:.2f}%) | "
                    f"AUC: {metrics['auc']:.3f}"
                )

    m = evaluate(model, val_loader, criterion, device)
    edge = (m["accuracy"] - m["baseline"]) * 100
    print(
        f"\nFinal Validation metrics:\n"
        f"  Loss        : {m['loss']:.4f}\n"
        f"  Accuracy    : {m['accuracy'] * 100:.2f}%\n"
        f"  Baseline    : {m['baseline'] * 100:.2f}%  (majority class; pos_rate={m['pos_rate'] * 100:.2f}%)\n"
        f"  Edge vs base: {edge:+.2f} pts  <-- the number that actually matters\n"
        f"  Precision   : {m['precision'] * 100:.2f}%\n"
        f"  Recall      : {m['recall'] * 100:.2f}%\n"
        f"  ROC AUC     : {m['auc']:.3f}  (0.5 = no skill)"
    )

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
