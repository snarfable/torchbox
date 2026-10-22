"""Unit tests for the finance wide-window RNN stock prediction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from torchbox.finance import (
    StockTimeSeriesDataset,
    WideWindowRNN,
    evaluate,
    fetch_stock_data,
    generate_synthetic_stock_data,
    plot_financial_data,
    train_epoch,
)
from torchbox.finance import (
    main as finance_main,
)


def test_generate_synthetic_stock_data() -> None:
    """Verifies that synthetic stock price generator returns valid arrays."""
    prices = generate_synthetic_stock_data(num_days=100, initial_price=50.0, seed=123)
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (100,)
    assert prices.dtype == np.float32
    assert (prices > 0.0).all()
    assert pytest.approx(prices[0]) == 50.0


def test_stock_time_series_dataset() -> None:
    """Verifies StockTimeSeriesDataset length, sample/target shapes, and normalization."""
    prices = generate_synthetic_stock_data(num_days=100, initial_price=100.0, seed=42)
    window_size = 20

    dataset = StockTimeSeriesDataset(prices, window_size=window_size, normalize=True)
    assert len(dataset) == 100 - window_size

    sample, target = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert sample.shape == (window_size, 1)
    assert target.shape == (1,)
    assert target.item() in (0.0, 1.0)

    # Test error when prices length <= window_size
    with pytest.raises(ValueError, match="must be greater than window_size"):
        StockTimeSeriesDataset(prices[:10], window_size=15)


@pytest.mark.parametrize("rnn_type", ["rnn", "lstm", "gru"])
def test_wide_window_rnn_forward(rnn_type: str) -> None:
    """Verifies WideWindowRNN forward pass output shape and probability output range [0, 1]."""
    window_size = 30
    batch_size = 4
    model = WideWindowRNN(input_size=1, hidden_size=32, window_size=window_size, rnn_type=rnn_type)

    # Test 3D tensor input: (batch_size, window_size, input_size)
    x3d = torch.randn(batch_size, window_size, 1)
    out3d = model(x3d)
    assert out3d.shape == (batch_size, 1)
    assert (out3d >= 0.0).all() and (out3d <= 1.0).all()

    # Test 2D tensor input: (batch_size, window_size)
    x2d = torch.randn(batch_size, window_size)
    out2d = model(x2d)
    assert out2d.shape == (batch_size, 1)
    assert (out2d >= 0.0).all() and (out2d <= 1.0).all()


def test_predict_boost_probability() -> None:
    """Verifies predict_boost_probability helper returns floats/arrays in [0, 1]."""
    window_size = 15
    model = WideWindowRNN(window_size=window_size)

    # 1D single sequence
    single_window = np.random.randn(window_size).astype(np.float32)
    prob_single = model.predict_boost_probability(single_window)
    assert isinstance(prob_single, float)
    assert 0.0 <= prob_single <= 1.0

    # Batch input
    batch_windows = np.random.randn(3, window_size, 1).astype(np.float32)
    probs_batch = model.predict_boost_probability(batch_windows)
    assert isinstance(probs_batch, np.ndarray)
    assert probs_batch.shape == (3,)
    assert (probs_batch >= 0.0).all() and (probs_batch <= 1.0).all()


def test_train_and_evaluate_epoch() -> None:
    """Verifies that a single training and evaluation epoch runs successfully."""
    prices = generate_synthetic_stock_data(num_days=150, seed=42)
    dataset = StockTimeSeriesDataset(prices, window_size=20)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    model = WideWindowRNN(window_size=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    device = torch.device("cpu")

    loss_epoch1 = train_epoch(model, loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, loader, criterion, device)

    assert isinstance(loss_epoch1, float)
    assert isinstance(val_loss, float)
    assert 0.0 <= val_acc <= 1.0


def test_fetch_stock_data_mocked() -> None:
    """Verifies fetch_stock_data correctly parses valid prices from Yahoo Finance payload."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "chart": {
            "result": [
                {
                    "indicators": {
                        "quote": [
                            {"close": [150.0, 152.5, None, 155.0]}
                        ]
                    }
                }
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        prices = fetch_stock_data("TEST")
        assert len(prices) == 3
        np.testing.assert_array_almost_equal(prices, [150.0, 152.5, 155.0])


def test_plot_financial_data(tmp_path: pytest.TempPath) -> None:
    """Verifies plot_financial_data creates and saves an image plot file."""
    prices = generate_synthetic_stock_data(num_days=50, seed=42)
    plot_path = str(tmp_path / "plot.png")
    plot_financial_data(prices, window_size=10, output_path=plot_path)
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_finance_main_cli(tmp_path: pytest.TempPath) -> None:
    """Verifies end-to-end main CLI execution with synthetic data, --dump-samples, and weight saving/loading."""
    model_path = str(tmp_path / "model.pt")
    plot_path = str(tmp_path / "samples.png")

    # Train, dump samples, and save model
    exit_code = finance_main([
        "--synthetic",
        "--window-size", "15",
        "--epochs", "2",
        "--dump-samples", plot_path,
        "--save-model", model_path,
    ])
    assert exit_code == 0
    assert (tmp_path / "samples.png").exists()

    # Load and evaluate
    exit_code_load = finance_main([
        "--synthetic",
        "--window-size", "15",
        "--epochs", "0",
        "--load-model", model_path,
    ])
    assert exit_code_load == 0
