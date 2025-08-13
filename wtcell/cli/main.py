"""Main CLI entry point for the wtcell package."""

import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel

from wtcell.cli.preprocess import preprocess_command
from wtcell.cli.train import train_command
from wtcell.cli.evaluate import evaluate_command
from wtcell.cli.predict import predict_command
from wtcell.utils.logging import setup_logging, get_logger

console = Console()
logger = get_logger(__name__)

@click.group()
@click.version_option(version="0.1.0", prog_name="wtcell")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Set logging level"
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Log file path"
)
def main(log_level: str, log_file: Optional[str]) -> None:
    """What The Cell - Cell segmentation for microscopy images using U-Net variants.
    
    This tool provides a complete pipeline for cell segmentation including:
    
    • Preprocessing: Generate masks from RLE annotations, K-Means clustering, or SIFT features
    • Training: Train U-Net models with various encoders (VGG16, MobileNetV2, scratch)
    • Evaluation: Compute metrics and generate visualizations
    • Prediction: Run inference on new images
    
    For more information, visit: https://github.com/neuefische/what-the-cell-and-where
    """
    # Setup logging
    setup_logging(level=log_level, log_file=log_file)
    
    # Display welcome message
    welcome_text = """
    [bold blue]What The Cell[/bold blue] - Cell Segmentation Tool
    
    [dim]Powered by U-Net and PyTorch[/dim]
    
    Use [bold]wtcell --help[/bold] to see available commands
    Use [bold]wtcell <command> --help[/bold] for command-specific help
    """
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))

@main.command()
@click.option(
    "--config",
    "-c",
    multiple=True,
    type=click.Path(exists=True),
    help="Configuration file(s) (YAML)"
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(["rle", "kmeans", "sift"]),
    default="rle",
    help="Mask source type"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for generated masks"
)
@click.option(
    "--cell-type",
    type=click.Choice(["astro", "cort", "shsy5y", "all"]),
    default="all",
    help="Cell type to process"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force regeneration of existing masks"
)
def preprocess(
    config: List[str],
    source: str,
    output_dir: Optional[str],
    cell_type: str,
    force: bool,
) -> None:
    """Preprocess images and generate segmentation masks.
    
    This command generates binary segmentation masks from microscopy images using
    one of three methods:
    
    • RLE: Decode Kaggle RLE annotations
    • K-Means: Apply clustering and Gaussian filtering
    • SIFT: Extract SIFT keypoints and cluster
    """
    try:
        preprocess_command(
            config_files=config,
            mask_source=source,
            output_dir=output_dir,
            cell_type=cell_type,
            force=force,
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@main.command()
@click.option(
    "--config",
    "-c",
    multiple=True,
    type=click.Path(exists=True),
    help="Configuration file(s) (YAML)"
)
@click.option(
    "--data-config",
    type=click.Path(exists=True),
    help="Data configuration file"
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    help="Model configuration file"
)
@click.option(
    "--train-config",
    type=click.Path(exists=True),
    help="Training configuration file"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for training artifacts"
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    help="Resume training from checkpoint"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without starting training"
)
def train(
    config: List[str],
    data_config: Optional[str],
    model_config: Optional[str],
    train_config: Optional[str],
    output_dir: Optional[str],
    resume: Optional[str],
    dry_run: bool,
) -> None:
    """Train a cell segmentation model.
    
    This command trains a U-Net model for cell segmentation using the specified
    configuration. The model can use different encoders (VGG16, MobileNetV2, or
    scratch) and various loss functions.
    """
    try:
        # Combine config files
        all_configs = list(config)
        if data_config:
            all_configs.append(data_config)
        if model_config:
            all_configs.append(model_config)
        if train_config:
            all_configs.append(train_config)
        
        train_command(
            config_files=all_configs,
            output_dir=output_dir,
            resume_checkpoint=resume,
            dry_run=dry_run,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Evaluation configuration file"
)
@click.option(
    "--checkpoint",
    "-ckpt",
    required=True,
    type=click.Path(exists=True),
    help="Model checkpoint path"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    help="Output directory for evaluation results"
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Prediction threshold for binary masks"
)
@click.option(
    "--save-predictions",
    is_flag=True,
    default=True,
    help="Save prediction masks"
)
@click.option(
    "--save-visualizations",
    is_flag=True,
    default=True,
    help="Save visualization plots"
)
def evaluate(
    config: Optional[str],
    checkpoint: str,
    output_dir: Optional[str],
    threshold: float,
    save_predictions: bool,
    save_visualizations: bool,
) -> None:
    """Evaluate a trained cell segmentation model.
    
    This command evaluates a trained model on test data, computing various metrics
    such as IoU, Dice coefficient, precision, recall, and F1 score. It also
    generates visualizations comparing predictions with ground truth.
    """
    try:
        evaluate_command(
            config_file=config,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            threshold=threshold,
            save_predictions=save_predictions,
            save_visualizations=save_visualizations,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Prediction configuration file"
)
@click.option(
    "--checkpoint",
    "-ckpt",
    required=True,
    type=click.Path(exists=True),
    help="Model checkpoint path"
)
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory or image file"
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for predictions"
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Prediction threshold for binary masks"
)
@click.option(
    "--save-overlays",
    is_flag=True,
    default=True,
    help="Save prediction overlays on original images"
)
@click.option(
    "--save-probabilities",
    is_flag=True,
    default=True,
    help="Save probability maps"
)
def predict(
    config: Optional[str],
    checkpoint: str,
    input: str,
    output: str,
    threshold: float,
    save_overlays: bool,
    save_probabilities: bool,
) -> None:
    """Run inference on new images.
    
    This command runs a trained model on new images to generate segmentation
    predictions. It can save binary masks, probability maps, and overlays.
    """
    try:
        predict_command(
            config_file=config,
            checkpoint_path=checkpoint,
            input_path=input,
            output_path=output,
            threshold=threshold,
            save_overlays=save_overlays,
            save_probabilities=save_probabilities,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@main.command()
def info() -> None:
    """Display package information and system status."""
    try:
        import torch
        import torchvision
        
        info_text = f"""
        [bold]Package Information[/bold]
        
        [bold]Version:[/bold] 0.1.0
        [bold]PyTorch:[/bold] {torch.__version__}
        [bold]Torchvision:[/bold] {torchvision.__version__}
        [bold]CUDA Available:[/bold] {torch.cuda.is_available()}
        
        [bold]System Information[/bold]
        
        [bold]Python:[/bold] {sys.version}
        [bold]Platform:[/bold] {sys.platform}
        """
        
        if torch.cuda.is_available():
            info_text += f"""
            [bold]CUDA Device:[/bold] {torch.cuda.get_device_name(0)}
            [bold]CUDA Memory:[/bold] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB
            """
        
        console.print(Panel(info_text, title="System Info", border_style="green"))
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main() 