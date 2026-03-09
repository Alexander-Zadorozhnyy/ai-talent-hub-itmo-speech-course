import pandas as pd
from pathlib import Path
import yaml

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class LogParser:
    def __init__(self, log_folder="logs"):
        self.log_folder = Path(log_folder)
        self.results = []

    def parse_hparams(self, hparams_file):
        """Parse hparams.yaml file."""
        with open(hparams_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def parse_metrics(self, metrics_file):
        """Parse metrics.csv file."""
        df = pd.read_csv(metrics_file)

        # Clean up column names (remove any extra spaces)
        df.columns = df.columns.str.strip()

        # Convert epoch to numeric if it's not
        if "epoch" in df.columns:
            df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

        return df

    def collect_all_runs(self):
        """Collect data from all model runs."""
        for model_dir in self.log_folder.glob("model_*"):
            if model_dir.is_dir():
                version_dir = model_dir / "version_0"
                if version_dir.exists():
                    hparams_file = version_dir / "hparams.yaml"
                    metrics_file = version_dir / "metrics.csv"

                    if hparams_file.exists() and metrics_file.exists():
                        try:
                            hparams = self.parse_hparams(hparams_file)
                            metrics = self.parse_metrics(metrics_file)

                            # Extract datetime from folder name
                            datetime_str = model_dir.name.replace("model_", "")

                            self.results.append(
                                {
                                    "model_dir": model_dir.name,
                                    "datetime": datetime_str,
                                    "hparams": hparams,
                                    "metrics": metrics,
                                    "n_mels": hparams.get("n_mels", None),
                                    "groups": hparams.get("groups", None),
                                    "lr": hparams.get("lr", None),
                                }
                            )
                            print(
                                f"Loaded: {model_dir.name} (n_mels={hparams.get('n_mels')}, groups={hparams.get('groups')})"
                            )
                        except Exception as e:
                            print(f"Error parsing {model_dir}: {e}")

        return self.results
