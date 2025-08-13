"""Preprocess CLI command implementation."""

from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from wtcell.masks import RLEMaskGenerator, KMeansMaskGenerator, SIFTMaskGenerator
from wtcell.utils.io import safe_path, ensure_dir, get_image_files
from wtcell.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)

def preprocess_command(
    config_files: List[str],
    mask_source: str,
    output_dir: Optional[str],
    cell_type: str,
    force: bool,
) -> None:
    """Execute preprocessing command.
    
    Args:
        config_files: List of configuration files
        mask_source: Type of mask source ('rle', 'kmeans', 'sift')
        output_dir: Output directory for masks
        cell_type: Cell type to process
        force: Force regeneration of existing masks
    """
    # Load configuration
    if config_files:
        config = load_config(config_files)
    else:
        config = get_default_config()
    
    # Override config with CLI arguments
    if output_dir:
        config.data.interim_dir = output_dir
    
    # Create mask generator
    mask_generator = create_mask_generator(mask_source, config)
    
    # Get image paths
    image_paths = get_image_paths(config, cell_type)
    
    if not image_paths:
        console.print("[yellow]No images found to process[/yellow]")
        return
    
    # Create output directory
    output_path = safe_path(config.data.interim_dir) / mask_source
    ensure_dir(output_path)
    
    # Process images
    process_images(
        image_paths=image_paths,
        mask_generator=mask_generator,
        output_path=output_path,
        force=force,
    )
    
    console.print(f"[green]Preprocessing completed! Generated {len(image_paths)} masks in {output_path}[/green]")

def load_config(config_files: List[str]) -> DictConfig:
    """Load configuration from files."""
    # Initialize Hydra
    hydra.initialize(version_base=None, config_path=".")
    
    # Compose configuration
    config = hydra.compose(config_name="data", overrides=[])
    
    # Override with additional config files
    for config_file in config_files:
        additional_config = hydra.compose(config_name=Path(config_file).stem)
        config = hydra.utils.merge_config(config, additional_config)
    
    return config

def get_default_config() -> DictConfig:
    """Get default configuration."""
    return {
        "data": {
            "raw_dir": "./data/raw",
            "interim_dir": "./data/interim",
            "images_dir": "./data/raw/train",
            "train_csv": "./data/raw/train.csv"
        }
    }

def create_mask_generator(mask_source: str, config: DictConfig) -> any:
    """Create appropriate mask generator."""
    if mask_source == "rle":
        return RLEMaskGenerator(config.masks.rle)
    elif mask_source == "kmeans":
        return KMeansMaskGenerator(config.masks.kmeans)
    elif mask_source == "sift":
        return SIFTMaskGenerator(config.masks.sift)
    else:
        raise ValueError(f"Unknown mask source: {mask_source}")

def get_image_paths(config: DictConfig, cell_type: str) -> List[Path]:
    """Get image paths to process."""
    images_dir = safe_path(config.data.images_dir)
    
    if not images_dir.exists():
        console.print(f"[red]Images directory not found: {images_dir}[/red]")
        return []
    
    # Get all image files
    image_files = get_image_files(images_dir)
    
    if cell_type != "all":
        # Filter by cell type if CSV is available
        csv_path = safe_path(config.data.train_csv)
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            cell_type_images = df[df["cell_type"] == cell_type]["id"].tolist()
            image_files = [f for f in image_files if f.stem in cell_type_images]
    
    return image_files

def process_images(
    image_paths: List[Path],
    mask_generator: any,
    output_path: Path,
    force: bool,
) -> None:
    """Process images and generate masks."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating masks...", total=len(image_paths))
        
        for i, image_path in enumerate(image_paths):
            progress.update(task, description=f"Processing {image_path.name} ({i+1}/{len(image_paths)})")
            
            # Generate output path
            mask_path = output_path / f"{image_path.stem}_mask.png"
            
            # Skip if mask exists and not forcing
            if mask_path.exists() and not force:
                logger.debug(f"Skipping {image_path.name} - mask already exists")
                continue
            
            try:
                # Generate mask
                mask = mask_generator.generate(image_path)
                
                # Save mask
                mask_generator.save_mask(mask, mask_path, format="png")
                
                logger.debug(f"Generated mask for {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                console.print(f"[red]Error processing {image_path.name}: {e}[/red]")
            
            progress.advance(task) 