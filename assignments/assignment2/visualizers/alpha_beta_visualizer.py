import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Literal, Tuple, Optional
import pandas as pd


class AlphaBetaHeatmapVisualizer:
    """
    A class to create heatmap visualizations comparing WER and CER across different
    alpha (language model weight) and beta (word insertion penalty) parameters.
    """

    def __init__(
        self,
        json_file_path: str,
        method: Literal["beam_lm"] | Literal["beam_lm_rescore"],
    ):
        """
        Initialize the visualizer with a JSON file containing alpha-beta comparison results.

        Args:
            json_file_path: Path to the JSON file containing the results
        """
        self.data = self._load_data(json_file_path)
        self.alphas = []
        self.betas = []
        self.wer_matrix = None
        self.cer_matrix = None
        self.time_matrix = None
        self.beam_width = None

        self.method = method

        self._extract_metrics()

    def _load_data(self, json_file_path: str) -> Dict:
        """Load and parse the JSON data."""
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_metrics(self):
        """Extract metrics from the loaded data and organize into matrices."""
        # Get unique alphas and betas
        alpha_set = set()
        beta_set = set()

        for result in self.data["results"]:
            alpha = result["params"]["alpha"]
            beta = result["params"]["beta"]
            alpha_set.add(alpha)
            beta_set.add(beta)
            if self.beam_width is None:
                self.beam_width = result["params"]["beam_width"]

        # Sort for consistent ordering
        self.alphas = sorted(alpha_set)
        self.betas = sorted(beta_set)

        # Create dictionaries for quick lookup
        alpha_to_idx = {alpha: i for i, alpha in enumerate(self.alphas)}
        beta_to_idx = {beta: i for i, beta in enumerate(self.betas)}

        # Initialize matrices
        self.wer_matrix = np.zeros((len(self.alphas), len(self.betas)))
        self.cer_matrix = np.zeros((len(self.alphas), len(self.betas)))
        self.time_matrix = np.zeros((len(self.alphas), len(self.betas)))

        # Fill matrices
        for result in self.data["results"]:
            alpha = result["params"]["alpha"]
            beta = result["params"]["beta"]
            wer = result["eval"][self.method]["metrics"]["wer"]
            cer = result["eval"][self.method]["metrics"]["cer"]
            inference_time = result["eval"][self.method]["avg_inference_time"]

            i = alpha_to_idx[alpha]
            j = beta_to_idx[beta]
            self.wer_matrix[i, j] = wer * 100
            self.cer_matrix[i, j] = cer * 100
            self.time_matrix[i, j] = inference_time

    def create_full_analysis(self, save_path: Optional[str] = None):
        """
        Create a comprehensive analysis with multiple visualizations.

        Args:
            save_path: Optional path to save the figure. If None, just displays.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. WER Heatmap
        sns.heatmap(
            self.wer_matrix,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            xticklabels=self.betas,
            yticklabels=self.alphas,
            cbar_kws={"label": "WER"},
            ax=axes[0],
        )
        axes[0].set_title("WER, %", fontsize=12)
        axes[0].set_ylabel("Alpha", fontsize=10)
        axes[0].set_xlabel("Beta", fontsize=10)

        # 2. CER Heatmap
        sns.heatmap(
            self.cer_matrix,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            xticklabels=self.betas,
            yticklabels=self.alphas,
            cbar_kws={"label": "CER"},
            ax=axes[1],
        )
        axes[1].set_title("CER, %", fontsize=12)
        axes[1].set_ylabel("Alpha", fontsize=10)
        axes[1].set_xlabel("Beta", fontsize=10)

        # 3. Best Parameters Table
        axes[2].axis("off")
        best_params = self.get_best_parameters()

        table_data = [
            [
                "Best WER",
                f"α={best_params['best_wer']['alpha']}",
                f"β={best_params['best_wer']['beta']}",
                f"{best_params['best_wer']['wer']:.4f}",
                f"{best_params['best_wer']['cer']:.4f}",
            ],
            [
                "Best CER",
                f"α={best_params['best_cer']['alpha']}",
                f"β={best_params['best_cer']['beta']}",
                f"{best_params['best_cer']['wer']:.4f}",
                f"{best_params['best_cer']['cer']:.4f}",
            ],
            [
                "Best Combined",
                f"α={best_params['best_combined']['alpha']}",
                f"β={best_params['best_combined']['beta']}",
                f"{best_params['best_combined']['wer']:.4f}",
                f"{best_params['best_combined']['cer']:.4f}",
            ],
            [
                "Fastest",
                f"α={best_params['fastest']['alpha']}",
                f"β={best_params['fastest']['beta']}",
                f"{best_params['fastest']['wer']:.4f}",
                f"{best_params['fastest']['cer']:.4f}",
            ],
        ]

        table = axes[2].table(
            cellText=table_data,
            colLabels=["Metric", "Alpha", "Beta", "WER", "CER"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        axes[2].set_title("Best Parameter Combinations", fontsize=12, pad=20)

        # Add overall title
        fig.suptitle(
            "Alpha-Beta Parameter Analysis", fontsize=16, fontweight="bold", y=1.02
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame with all metrics.

        Returns:
            Pandas DataFrame with all alpha-beta combinations and metrics
        """
        data = []
        for i, alpha in enumerate(self.alphas):
            for j, beta in enumerate(self.betas):
                data.append(
                    {
                        "Alpha": alpha,
                        "Beta": beta,
                        "WER": self.wer_matrix[i, j],
                        "CER": self.cer_matrix[i, j],
                        "Inference_Time": self.time_matrix[i, j],
                    }
                )

        df = pd.DataFrame(data)

        return df

    def get_best_parameters(self) -> Dict:
        """
        Find the best parameter combinations for different metrics.

        Returns:
            Dictionary with best parameter combinations
        """
        wer_min_idx = np.unravel_index(
            np.argmin(self.wer_matrix), self.wer_matrix.shape
        )
        cer_min_idx = np.unravel_index(
            np.argmin(self.cer_matrix), self.cer_matrix.shape
        )

        wer_norm = (self.wer_matrix - np.min(self.wer_matrix)) / (
            np.max(self.wer_matrix) - np.min(self.wer_matrix)
        )
        cer_norm = (self.cer_matrix - np.min(self.cer_matrix)) / (
            np.max(self.cer_matrix) - np.min(self.cer_matrix)
        )
        combined_score = wer_norm + cer_norm
        combined_min_idx = np.unravel_index(
            np.argmin(combined_score), combined_score.shape
        )

        time_min_idx = np.unravel_index(
            np.argmin(self.time_matrix), self.time_matrix.shape
        )

        return {
            "best_wer": {
                "alpha": self.alphas[wer_min_idx[0]],
                "beta": self.betas[wer_min_idx[1]],
                "wer": self.wer_matrix[wer_min_idx[0], wer_min_idx[1]],
                "cer": self.cer_matrix[wer_min_idx[0], wer_min_idx[1]],
            },
            "best_cer": {
                "alpha": self.alphas[cer_min_idx[0]],
                "beta": self.betas[cer_min_idx[1]],
                "wer": self.wer_matrix[cer_min_idx[0], cer_min_idx[1]],
                "cer": self.cer_matrix[cer_min_idx[0], cer_min_idx[1]],
            },
            "best_combined": {
                "alpha": self.alphas[combined_min_idx[0]],
                "beta": self.betas[combined_min_idx[1]],
                "wer": self.wer_matrix[combined_min_idx[0], combined_min_idx[1]],
                "cer": self.cer_matrix[combined_min_idx[0], combined_min_idx[1]],
                "combined_score": combined_score[
                    combined_min_idx[0], combined_min_idx[1]
                ],
            },
            "fastest": {
                "alpha": self.alphas[time_min_idx[0]],
                "beta": self.betas[time_min_idx[1]],
                "time": self.time_matrix[time_min_idx[0], time_min_idx[1]],
                "wer": self.wer_matrix[time_min_idx[0], time_min_idx[1]],
                "cer": self.cer_matrix[time_min_idx[0], time_min_idx[1]],
            },
        }


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = AlphaBetaHeatmapVisualizer(
        "./results/beam_lm_rescore_comparison.json",
        method="beam_lm_rescore",
    )

    # Get summary statistics
    print("Summary Statistics:")
    print(visualizer.get_summary_dataframe().head(10))
    print("\n")

    print("Best Parameters:")
    best = visualizer.get_best_parameters()
    for metric, params in best.items():
        print(f"  {metric}: {params}")
    print("\n")

    # Create full analysis with all plots
    fig_full = visualizer.create_full_analysis(
        save_path="./imgs/alpha_beta_lm_rescore_analysis.png"
    )
    plt.show()
