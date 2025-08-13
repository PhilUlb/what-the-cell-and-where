"""Predict CLI command implementation."""

def predict_command(config_file, checkpoint_path, input_path, output_path, threshold, save_overlays, save_probabilities):
    """Execute prediction command."""
    print(f"Predicting with checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Threshold: {threshold}")
    print("Prediction command not yet implemented") 