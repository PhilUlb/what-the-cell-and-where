# What The Cell - Cell Segmentation Tool

A production-grade Python package for cell segmentation in microscopy images using U-Net variants, trained on the Kaggle Sartorius dataset.

## Features

- **Multiple Mask Sources**: Generate masks from RLE annotations, K-Means clustering, or SIFT features
- **Flexible U-Net Architecture**: Support for VGG16, MobileNetV2, and scratch encoders
- **Production Ready**: Type hints, comprehensive testing, logging, and configuration management
- **GPU Support**: CUDA-ready with automatic fallback to CPU
- **CLI Interface**: Simple command-line tools for all operations

## Quick Start

### 1. Installation

```bash
# Install with GPU support (recommended)
poetry install --with gpu

# Or install CPU-only version
poetry install --without gpu

# Install development dependencies
poetry install --with dev
```

### 2. Setup

```bash
# Install pre-commit hooks and setup development environment
make setup

# Or manually
poetry install --with dev
pre-commit install
```

### 3. Download Dataset

```bash
# Set Kaggle credentials (optional)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download dataset
poetry run wtcell download-dataset
```

### 4. Generate Masks

```bash
# Generate masks from RLE annotations
poetry run wtcell preprocess --source rle --config configs/data.yaml

# Generate masks using K-Means clustering
poetry run wtcell preprocess --source kmeans --config configs/data.yaml

# Generate masks using SIFT features
poetry run wtcell preprocess --source sift --config configs/data.yaml
```

### 5. Train Model

```bash
# Train U-Net from scratch
poetry run wtcell train \
    --config configs/data.yaml \
    --config configs/model.yaml \
    --config configs/train.yaml

# Train with specific encoder
poetry run wtcell train \
    --config configs/data.yaml \
    --config configs/model.yaml \
    --config configs/train.yaml \
    --model-encoder vgg16
```

### 6. Evaluate Model

```bash
# Evaluate trained model
poetry run wtcell evaluate \
    --config configs/eval.yaml \
    --checkpoint runs/latest/best.ckpt
```

### 7. Run Inference

```bash
# Predict on new images
poetry run wtcell predict \
    --config configs/predict.yaml \
    --checkpoint runs/latest/best.ckpt \
    --input data/test_images \
    --output data/predictions
```

## Project Structure

```
wtcell/
├── configs/              # Configuration files
├── wtcell/              # Main package
│   ├── cli/            # Command-line interface
│   ├── data/           # Data loading and preprocessing
│   ├── masks/          # Mask generation algorithms
│   ├── models/         # U-Net and encoder architectures
│   ├── training/       # Training loops and losses
│   ├── eval/           # Evaluation and metrics
│   └── utils/          # Utilities and helpers
├── tests/              # Test suite
├── scripts/            # Utility scripts
└── docs/               # Documentation
```

## Configuration

The package uses YAML configuration files for all settings:

- `configs/data.yaml` - Data paths, augmentation, and loading settings
- `configs/model.yaml` - Model architecture and encoder settings
- `configs/train.yaml` - Training hyperparameters and settings
- `configs/eval.yaml` - Evaluation metrics and visualization settings
- `configs/predict.yaml` - Inference and output settings

## Mask Generation Methods

### 1. RLE Masks (Ground Truth)
- Decode Kaggle RLE annotations
- Most accurate but requires annotation data
- Fastest generation method

### 2. K-Means Clustering
- Apply K-Means clustering to pixel values
- Follow with Gaussian filtering and thresholding
- Good for images with distinct intensity regions

### 3. SIFT Features
- Extract SIFT keypoints from images
- Apply clustering to keypoint-enhanced images
- Most computationally intensive but can capture complex patterns

## Model Architecture

The U-Net implementation supports:

- **Encoders**: VGG16, MobileNetV2, or scratch
- **Loss Functions**: BCE, Dice, BCE+Dice, Focal, IoU
- **Optimizers**: Adam, AdamW, SGD
- **Schedulers**: Cosine, Step, Plateau, OneCycle

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=wtcell

# Run specific test file
poetry run pytest tests/test_rle.py
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make typecheck

# Run all quality checks
make ci
```

### Building Documentation

```bash
# Build docs
make docs

# Serve docs locally
make docs-serve
```

## Docker

```bash
# Build image
make docker-build

# Run with GPU
make docker-run

# Run CPU-only
make docker-run-cpu
```

## Environment Variables

```bash
# Kaggle API (optional)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Training settings
export SEED=42
export NUM_WORKERS=4
export BATCH_SIZE=8

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export MIXED_PRECISION=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{wtcell2024,
  title={What The Cell - Cell Segmentation Tool},
  author={neuefische GmbH},
  year={2024},
  url={https://github.com/neuefische/what-the-cell-and-where}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the test examples

## Acknowledgments

- Original research by neuefische GmbH Data Science Bootcamp
- U-Net architecture by Ronneberger et al.
- PyTorch and PyTorch Lightning communities

