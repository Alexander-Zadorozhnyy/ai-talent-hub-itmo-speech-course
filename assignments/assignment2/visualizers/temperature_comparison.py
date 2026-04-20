import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import pandas as pd


class TemperatureComparisonVisualizer:
    """
    A class to compare temperature effects on WER and CER between greedy decoding
    and shallow fusion (language model) methods.

    This class extracts metrics from two JSON files (greedy and shallow fusion) and
    creates comparison plots for temperature vs WER and temperature vs CER.
    """

    def __init__(self, greedy_json_path: str, shallow_fusion_json_path: str):
        """
        Initialize the visualizer with two JSON files containing temperature comparison results.

        Args:
            greedy_json_path: Path to the JSON file with greedy decoding results
            shallow_fusion_json_path: Path to the JSON file with shallow fusion results
        """
        self.greedy_data = self._load_data(greedy_json_path)
        self.shallow_fusion_data = self._load_data(shallow_fusion_json_path)

        self.temperatures = []
        self.greedy_wer = []
        self.greedy_cer = []
        self.greedy_times = []

        self.shallow_wer = []
        self.shallow_cer = []
        self.shallow_times = []

        self._extract_metrics()

    def _load_data(self, json_file_path: str) -> Dict:
        """Load and parse the JSON data."""
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_metrics(self):
        """Extract metrics from both datasets."""
        # Extract from greedy data
        for result in self.greedy_data["results"]:
            temp = result["params"]["temperature"]
            method = list(result["eval"].keys())[0]
            wer = result["eval"][method]["metrics"]["wer"]
            cer = result["eval"][method]["metrics"]["cer"]
            inference_time = result["eval"][method]["avg_inference_time"]

            self.temperatures.append(temp)
            self.greedy_wer.append(wer)
            self.greedy_cer.append(cer)
            self.greedy_times.append(inference_time)

        # Sort by temperature
        sorted_indices = sorted(
            range(len(self.temperatures)), key=lambda i: self.temperatures[i]
        )
        self.temperatures = [self.temperatures[i] for i in sorted_indices]
        self.greedy_wer = [self.greedy_wer[i] for i in sorted_indices]
        self.greedy_cer = [self.greedy_cer[i] for i in sorted_indices]
        self.greedy_times = [self.greedy_times[i] for i in sorted_indices]

        # Extract from shallow fusion data
        for result in self.shallow_fusion_data["results"]:
            temp = result["params"]["temperature"]
            method = list(result["eval"].keys())[0]
            wer = result["eval"][method]["metrics"]["wer"]
            cer = result["eval"][method]["metrics"]["cer"]
            inference_time = result["eval"][method]["avg_inference_time"]

            self.shallow_wer.append(wer)
            self.shallow_cer.append(cer)
            self.shallow_times.append(inference_time)

        # Sort shallow fusion data by temperature
        shallow_sorted = sorted(
            zip(
                self.temperatures,
                self.shallow_wer,
                self.shallow_cer,
                self.shallow_times,
            ),
            key=lambda x: x[0],
        )
        self.temperatures = [t for t, _, _, _ in shallow_sorted]
        self.shallow_wer = [w for _, w, _, _ in shallow_sorted]
        self.shallow_cer = [c for _, _, c, _ in shallow_sorted]
        self.shallow_times = [t for _, _, _, t in shallow_sorted]

        # Ensure greedy data is sorted the same way
        greedy_sorted = sorted(
            zip(self.temperatures, self.greedy_wer, self.greedy_cer, self.greedy_times),
            key=lambda x: x[0],
        )
        self.greedy_wer = [w for _, w, _, _ in greedy_sorted]
        self.greedy_cer = [c for _, _, c, _ in greedy_sorted]
        self.greedy_times = [t for _, _, _, t in greedy_sorted]

    def plot_temperature_vs_wer(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a line plot comparing temperature vs Word Error Rate (WER).

        Args:
            figsize: Figure size (width, height) in inches
            title: Custom title for the plot
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot lines
        ax.plot(
            self.temperatures,
            self.greedy_wer,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Greedy Decoding",
            color="blue",
        )

        ax.plot(
            self.temperatures,
            self.shallow_wer,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Shallow Fusion (LM)",
            color="orange",
        )

        # Add data labels
        for i, temp in enumerate(self.temperatures):
            ax.annotate(
                f"{self.greedy_wer[i]:.3f}",
                (temp, self.greedy_wer[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="blue",
            )

            ax.annotate(
                f"{self.shallow_wer[i]:.3f}",
                (temp, self.shallow_wer[i]),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="orange",
            )

        # Customize plot
        if title is None:
            title = "Temperature vs Word Error Rate (WER)"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Word Error Rate (WER)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set x-axis to show all temperature values
        ax.set_xticks(self.temperatures)
        ax.set_xticklabels([f"{t:.1f}" for t in self.temperatures])

        # Add some padding to y-axis
        y_min = min(min(self.greedy_wer), min(self.shallow_wer))
        y_max = max(max(self.greedy_wer), max(self.shallow_wer))
        y_padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig

    def plot_temperature_vs_cer(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a line plot comparing temperature vs Character Error Rate (CER).

        Args:
            figsize: Figure size (width, height) in inches
            title: Custom title for the plot
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot lines
        ax.plot(
            self.temperatures,
            self.greedy_cer,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Greedy Decoding",
            color="blue",
        )

        ax.plot(
            self.temperatures,
            self.shallow_cer,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Shallow Fusion (LM)",
            color="orange",
        )

        # Add data labels
        for i, temp in enumerate(self.temperatures):
            ax.annotate(
                f"{self.greedy_cer[i]:.3f}",
                (temp, self.greedy_cer[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="blue",
            )

            ax.annotate(
                f"{self.shallow_cer[i]:.3f}",
                (temp, self.shallow_cer[i]),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="orange",
            )

        # Customize plot
        if title is None:
            title = "Temperature vs Character Error Rate (CER)"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set x-axis to show all temperature values
        ax.set_xticks(self.temperatures)
        ax.set_xticklabels([f"{t:.1f}" for t in self.temperatures])

        # Add some padding to y-axis
        y_min = min(min(self.greedy_cer), min(self.shallow_cer))
        y_max = max(max(self.greedy_cer), max(self.shallow_cer))
        y_padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig

    def plot_combined_comparison(
        self, figsize: Tuple[int, int] = (12, 10), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a combined figure with both WER and CER comparisons.

        Args:
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # WER Plot
        ax1 = axes[0]
        ax1.plot(
            self.temperatures,
            self.greedy_wer,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Greedy Decoding",
            color="blue",
        )
        ax1.plot(
            self.temperatures,
            self.shallow_wer,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Shallow Fusion (LM)",
            color="orange",
        )

        # Add data labels for WER
        for i, temp in enumerate(self.temperatures):
            ax1.annotate(
                f"{self.greedy_wer[i]:.4f}",
                (temp, self.greedy_wer[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="blue",
            )

            ax1.annotate(
                f"{self.shallow_wer[i]:.4f}",
                (temp, self.shallow_wer[i]),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=8,
                color="orange",
            )

        ax1.set_title(
            "Temperature vs Word Error Rate (WER)", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Temperature", fontsize=10)
        ax1.set_ylabel("Word Error Rate (WER)", fontsize=10)
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.set_xticks(self.temperatures)
        ax1.set_xticklabels([f"{t:.1f}" for t in self.temperatures])

        # CER Plot
        ax2 = axes[1]
        ax2.plot(
            self.temperatures,
            self.greedy_cer,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Greedy Decoding",
            color="blue",
        )
        ax2.plot(
            self.temperatures,
            self.shallow_cer,
            marker="s",
            linewidth=2,
            markersize=8,
            label="Shallow Fusion (LM)",
            color="orange",
        )

        # Add data labels for CER
        for i, temp in enumerate(self.temperatures):
            ax2.annotate(
                f"{self.greedy_cer[i]:.4f}",
                (temp, self.greedy_cer[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="blue",
            )

            ax2.annotate(
                f"{self.shallow_cer[i]:.4f}",
                (temp, self.shallow_cer[i]),
                textcoords="offset points",
                xytext=(0, -12),
                ha="center",
                fontsize=8,
                color="orange",
            )

        ax2.set_title(
            "Temperature vs Character Error Rate (CER)", fontsize=12, fontweight="bold"
        )
        ax2.set_xlabel("Temperature", fontsize=10)
        ax2.set_ylabel("Character Error Rate (CER)", fontsize=10)
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.set_xticks(self.temperatures)
        ax2.set_xticklabels([f"{t:.1f}" for t in self.temperatures])

        plt.suptitle(
            "Temperature Effect on Speech Recognition Performance",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame with all metrics and improvements.

        Returns:
            Pandas DataFrame with summary statistics
        """

        df = pd.DataFrame(
            {
                "Temperature": self.temperatures,
                "Greedy_WER": self.greedy_wer,
                "Greedy_CER": self.greedy_cer,
                "Shallow_WER": self.shallow_wer,
                "Shallow_CER": self.shallow_cer,
                "Greedy_Time": self.greedy_times,
                "Shallow_Time": self.shallow_times,
            }
        )

        return df


# Example usage
if __name__ == "__main__":
    # Initialize visualizer with both JSON files
    visualizer = TemperatureComparisonVisualizer(
        "./results/earning_test_greedy_temperature.json",
        "./results/earning_test_shallow_fusion_temperature.json",
    )

    # Get summary DataFrame
    print("Temperature Comparison Summary:")
    df = visualizer.get_summary_dataframe()
    print(df.to_string(index=False))
    print("\n")

    # Create individual plots
    fig_wer = visualizer.plot_temperature_vs_wer()
    plt.show()

    fig_cer = visualizer.plot_temperature_vs_cer()
    plt.show()

    # Create combined plot
    fig_combined = visualizer.plot_combined_comparison()
    plt.show()

    # Create full analysis with all plots and save
    visualizer.plot_combined_comparison(save_path="./imgs/temperature_comparison")
