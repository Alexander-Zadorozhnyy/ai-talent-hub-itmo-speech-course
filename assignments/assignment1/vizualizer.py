import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, results, plot_path: str):
        self.plot_path = plot_path
        self.results = results
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up matplotlib style."""
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except:
            plt.style.use("ggplot")
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_n_mels_comparison(self, save_path="n_mels_comparison.png"):
        save_path = f"{self.plot_path}/{save_path}"
        """Plot comparisons for different n_mels values."""
        # Filter runs with n_mels parameter
        n_mels_runs = [r for r in self.results if r["n_mels"] is not None]

        if not n_mels_runs:
            print("No runs with n_mels parameter found")
            return

        print(f"\nPlotting n_mels comparison with {len(n_mels_runs)} runs...")

        # Group by n_mels
        n_mels_values = sorted(set(r["n_mels"] for r in n_mels_runs))

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle("Effect of n_mels Parameter on Model Performance", fontsize=16)

        # Store data for scatter plot
        scatter_data = []

        for i, n_mels in enumerate(n_mels_values):
            runs = [r for r in n_mels_runs if r["n_mels"] == n_mels]
            color = self.colors[i % len(self.colors)]
            label_added = False

            for run_idx, run in enumerate(runs):
                metrics = run["metrics"]

                if metrics.empty:
                    continue

                # Plot training loss
                if (
                    "train_loss" in metrics.columns
                    and metrics["train_loss"].notna().any()
                ):
                    valid_data = metrics[metrics["train_loss"].notna()]
                    if not valid_data.empty:
                        axes[0, 0].plot(
                            valid_data["epoch"],
                            valid_data["train_loss"],
                            color=color,
                            alpha=0.7,
                            linewidth=2,
                            label=f"n_mels={n_mels}" if not label_added else "",
                        )
                        label_added = True

                # Plot validation loss
                if (
                    "valid_loss" in metrics.columns
                    and metrics["valid_loss"].notna().any()
                ):
                    valid_data = metrics[metrics["valid_loss"].notna()]
                    if not valid_data.empty:
                        axes[0, 1].plot(
                            valid_data["epoch"],
                            valid_data["valid_loss"],
                            color=color,
                            alpha=0.7,
                            linewidth=2,
                            label=f"n_mels={n_mels}" if not label_added else "",
                        )

                # Plot validation accuracy
                if (
                    "valid_accuracy" in metrics.columns
                    and metrics["valid_accuracy"].notna().any()
                ):
                    valid_data = metrics[metrics["valid_accuracy"].notna()]
                    if not valid_data.empty:
                        axes[1, 0].plot(
                            valid_data["epoch"],
                            valid_data["valid_accuracy"],
                            color=color,
                            alpha=0.7,
                            linewidth=2,
                            label=f"n_mels={n_mels}" if not label_added else "",
                        )

                # Collect test accuracy
                if (
                    "test_accuracy" in metrics.columns
                    and metrics["test_accuracy"].notna().any()
                ):
                    test_acc = metrics["test_accuracy"].dropna().iloc[-1]
                    scatter_data.append({"n_mels": n_mels, "test_accuracy": test_acc})

        # Set labels and legends
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Train Loss")
        axes[0, 0].set_title("Training Loss vs Epoch")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Validation Loss")
        axes[0, 1].set_title("Validation Loss vs Epoch")
        axes[0, 1].legend(loc="upper right", fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Validation Accuracy")
        axes[1, 0].set_title("Validation Accuracy vs Epoch")
        axes[1, 0].legend(loc="lower right", fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot test accuracy vs n_mels
        if scatter_data:
            df_scatter = pd.DataFrame(scatter_data)

            n_mels_list = df_scatter["n_mels"].values
            acc_list = df_scatter["test_accuracy"].values

            box_data = []
            positions = []
            for n_mels in sorted(df_scatter["n_mels"].unique()):
                data = df_scatter[df_scatter["n_mels"] == n_mels][
                    "test_accuracy"
                ].values
                box_data.append(data)
                positions.append(n_mels)

            if box_data:
                bp = axes[1, 1].boxplot(
                    box_data, positions=positions, widths=5, patch_artist=True
                )
                for patch, color in zip(
                    bp["boxes"],
                    [self.colors[i % len(self.colors)] for i in range(len(box_data))],
                ):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Add individual points
                for n_mels in sorted(df_scatter["n_mels"].unique()):
                    data = df_scatter[df_scatter["n_mels"] == n_mels]["test_accuracy"]
                    x = np.random.normal(n_mels, 1, len(data))
                    axes[1, 1].scatter(x, data, color="black", alpha=0.5, s=50)

            axes[1, 1].set_xlabel("n_mels")
            axes[1, 1].set_ylabel("Test Accuracy")
            axes[1, 1].set_title("Test Accuracy vs n_mels")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(n_mels_values)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
        print(f"Saved n_mels comparison plot to {save_path}")

    def plot_groups_comparison(self, save_path="groups_comparison.png"):
        """Plot comparisons for different groups parameters."""
        save_path = f"{self.plot_path}/{save_path}"
        # Filter runs with groups parameter
        groups_runs = [r for r in self.results if r["groups"] is not None]

        if not groups_runs:
            print("No runs with groups parameter found")
            return

        print(f"\nPlotting groups comparison with {len(groups_runs)} runs...")

        # Group by groups
        groups_values = sorted(set(r["groups"] for r in groups_runs))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Effect of groups Parameter on Model Performance and Efficiency",
            fontsize=16,
        )

        groups_data = []

        for i, groups in enumerate(groups_values):
            runs = [r for r in groups_runs if r["groups"] == groups]
            color = self.colors[i % len(self.colors)]
            label_added = False

            for run in runs:
                metrics = run["metrics"]

                # Skip if metrics is empty
                if metrics.empty:
                    continue

                # Plot epoch time
                if (
                    "epoch_time" in metrics.columns
                    and metrics["epoch_time"].notna().any()
                ):
                    valid_data = metrics[metrics["epoch_time"].notna()]
                    if not valid_data.empty:
                        axes[0, 0].plot(
                            valid_data["epoch"],
                            valid_data["epoch_time"],
                            color=color,
                            alpha=0.7,
                            marker="o",
                            markersize=4,
                            linewidth=2,
                            label=f"groups={groups}" if not label_added else "",
                        )

                # Plot validation accuracy
                if (
                    "valid_accuracy" in metrics.columns
                    and metrics["valid_accuracy"].notna().any()
                ):
                    valid_data = metrics[metrics["valid_accuracy"].notna()]
                    if not valid_data.empty:
                        axes[0, 1].plot(
                            valid_data["epoch"],
                            valid_data["valid_accuracy"],
                            color=color,
                            alpha=0.7,
                            marker="s",
                            markersize=4,
                            linewidth=2,
                            label=f"groups={groups}" if not label_added else "",
                        )
                        label_added = True

                # Collect final metrics
                final_metrics = {
                    "groups": groups,
                    "epoch_time": metrics["epoch_time"].mean()
                    if "epoch_time" in metrics.columns
                    and metrics["epoch_time"].notna().any()
                    else None,
                    "params": metrics["params"].dropna().iloc[-1]
                    if "params" in metrics.columns and metrics["params"].notna().any()
                    else None,
                    "flops": metrics["flops"].dropna().iloc[-1]
                    if "flops" in metrics.columns and metrics["flops"].notna().any()
                    else None,
                    "test_accuracy": metrics["test_accuracy"].dropna().iloc[-1]
                    if "test_accuracy" in metrics.columns
                    and metrics["test_accuracy"].notna().any()
                    else None,
                }
                groups_data.append(final_metrics)

        # Set labels for first row
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Epoch Time (s)")
        axes[0, 0].set_title("Training Time per Epoch")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Validation Accuracy")
        axes[0, 1].set_title("Validation Accuracy vs Epoch")
        axes[0, 1].legend(loc="lower right", fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot groups comparisons
        if groups_data:
            df = pd.DataFrame(groups_data)

            # Average epoch time vs groups
            if df["epoch_time"].notna().any():
                epoch_time_means = df.groupby("groups")["epoch_time"].mean()
                epoch_time_stds = df.groupby("groups")["epoch_time"].std().fillna(0)

                axes[1, 0].bar(
                    epoch_time_means.index.astype(str),
                    epoch_time_means.values,
                    yerr=epoch_time_stds.values,
                    capsize=5,
                    color=[
                        self.colors[i % len(self.colors)]
                        for i in range(len(epoch_time_means))
                    ],
                    alpha=0.7,
                )
                axes[1, 0].set_xlabel("groups")
                axes[1, 0].set_ylabel("Average Epoch Time (s)")
                axes[1, 0].set_title("Average Training Time vs groups")
                axes[1, 0].grid(True, alpha=0.3, axis="y")

            # Parameters vs groups
            if df["params"].notna().any():
                params_means = df.groupby("groups")["params"].mean()
                params_stds = df.groupby("groups")["params"].std().fillna(0)

                axes[1, 1].bar(
                    params_means.index.astype(str),
                    params_means.values,
                    yerr=params_stds.values,
                    capsize=5,
                    color=[
                        self.colors[i % len(self.colors)]
                        for i in range(len(params_means))
                    ],
                    alpha=0.7,
                )
                axes[1, 1].set_xlabel("groups")
                axes[1, 1].set_ylabel("Number of Parameters")
                axes[1, 1].set_title("Model Parameters vs groups")
                axes[1, 1].grid(True, alpha=0.3, axis="y")
                axes[1, 1].ticklabel_format(
                    style="scientific", axis="y", scilimits=(4, 4)
                )

            # FLOPs vs groups
            if df["flops"].notna().any():
                flops_means = df.groupby("groups")["flops"].mean()
                flops_stds = df.groupby("groups")["flops"].std().fillna(0)

                axes[1, 2].bar(
                    flops_means.index.astype(str),
                    flops_means.values,
                    yerr=flops_stds.values,
                    capsize=5,
                    color=[
                        self.colors[i % len(self.colors)]
                        for i in range(len(flops_means))
                    ],
                    alpha=0.7,
                )
                axes[1, 2].set_xlabel("groups")
                axes[1, 2].set_ylabel("FLOPs")
                axes[1, 2].set_title("FLOPs vs groups")
                axes[1, 2].grid(True, alpha=0.3, axis="y")
                axes[1, 2].ticklabel_format(
                    style="scientific", axis="y", scilimits=(4, 4)
                )

            # Test accuracy vs groups
            if df["test_accuracy"].notna().any():
                acc_means = df.groupby("groups")["test_accuracy"].mean()
                acc_stds = df.groupby("groups")["test_accuracy"].std().fillna(0)

                axes[0, 2].errorbar(
                    acc_means.index.astype(str),
                    acc_means.values,
                    yerr=acc_stds.values,
                    fmt="o-",
                    capsize=5,
                    color="red",
                    linewidth=2,
                    markersize=8,
                )
                axes[0, 2].set_xlabel("groups")
                axes[0, 2].set_ylabel("Test Accuracy")
                axes[0, 2].set_title("Test Accuracy vs groups")
                axes[0, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved groups comparison plot to {save_path}")
