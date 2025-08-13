"""Command-line interface for the wtcell package."""

from wtcell.cli.main import main
from wtcell.cli.preprocess import preprocess_command
from wtcell.cli.train import train_command
from wtcell.cli.evaluate import evaluate_command
from wtcell.cli.predict import predict_command

__all__ = [
    "main",
    "preprocess_command",
    "train_command",
    "evaluate_command",
    "predict_command",
] 