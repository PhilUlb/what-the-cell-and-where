"""Logging utilities for the wtcell package."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler
install()

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rich_console: bool = True,
) -> None:
    """Setup logging configuration for the package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        rich_console: Whether to use rich console formatting
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if rich_console:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
        )
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = get_logger(__name__)
    logger.info(f"Random seed set to {seed}") 