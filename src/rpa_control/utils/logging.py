"""Experiment logging utilities."""
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class ExperimentLogger:
    """Logger for tracking experiments with all settings and outputs."""

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """Initialize experiment logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this experiment (default: auto-generated timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Generate experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = timestamp

        # Create experiment directory
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # File paths
        self.config_path = self.experiment_dir / "config.json"
        self.history_path = self.experiment_dir / "history.json"
        self.console_path = self.experiment_dir / "console.log"
        self.results_path = self.experiment_dir / "results.txt"

        # Console output capture
        self.console_log = open(self.console_path, 'w')
        self.original_stdout = sys.stdout

    def start_capture(self):
        """Start capturing console output."""
        sys.stdout = TeeOutput(self.original_stdout, self.console_log)

    def stop_capture(self):
        """Stop capturing console output."""
        sys.stdout = self.original_stdout
        self.console_log.close()

    def log_config(self, config: Dict[str, Any]):
        """Save experiment configuration.

        Args:
            config: Dictionary of configuration parameters
        """
        # Convert non-serializable types
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_config[key] = value
            elif isinstance(value, Path):
                serializable_config[key] = str(value)
            elif torch.is_tensor(value):
                serializable_config[key] = value.tolist()
            else:
                serializable_config[key] = str(value)

        with open(self.config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)

        print(f"Configuration saved to: {self.config_path}")

    def log_history(self, history: Dict[str, Any]):
        """Save training history.

        Args:
            history: Dictionary of training metrics (loss, reward, etc.)
        """
        # Convert tensors to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list):
                # Check if list contains tensors
                if value and torch.is_tensor(value[0]):
                    serializable_history[key] = [v.item() if torch.is_tensor(v) else v for v in value]
                else:
                    serializable_history[key] = value
            elif torch.is_tensor(value):
                if value.numel() == 1:
                    serializable_history[key] = value.item()
                else:
                    serializable_history[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool)):
                serializable_history[key] = value
            else:
                serializable_history[key] = str(value)

        with open(self.history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        print(f"Training history saved to: {self.history_path}")

    def log_results(self, summary_dict: Dict[str, Any], controller_summary: str = None):
        """Save experiment results (summary + controller).

        Args:
            summary_dict: Dictionary of summary statistics
            controller_summary: Optional human-readable controller equations
        """
        with open(self.results_path, 'w') as f:
            f.write("Experiment Results\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write summary
            f.write("Summary\n")
            f.write("-" * 60 + "\n")
            for key, value in summary_dict.items():
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        f.write(f"{key}: {value.item()}\n")
                    else:
                        f.write(f"{key}: {value.tolist()}\n")
                else:
                    f.write(f"{key}: {value}\n")

            # Write controller if provided
            if controller_summary:
                f.write("\n")
                f.write("Trained Controller\n")
                f.write("-" * 60 + "\n")
                f.write(controller_summary)
                f.write("\n")

        print(f"Results saved to: {self.results_path}")

    def get_experiment_path(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir

    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()


class TeeOutput:
    """Redirect output to both console and file."""

    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, data):
        for output in self.outputs:
            output.write(data)
            output.flush()

    def flush(self):
        for output in self.outputs:
            output.flush()
