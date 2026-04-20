from log_parser import LogParser
from vizualizer import Visualizer


def main():
    plot_n_mels()
    plot_groups()

    print("\nAnalysis complete! Check the 'plots' directory for results.")


def plot_n_mels():
    parser = LogParser(log_folder="n_mels_comp_logs")
    results = parser.collect_all_runs()

    print(f"\nFound {len(results)} model runs")

    if not results:
        print("No results found. Check your logs folder structure.")
        return

    viz = Visualizer(results, plot_path="imgs")
    viz.plot_n_mels_comparison()


def plot_groups():
    parser = LogParser(log_folder="group_comp_logs")
    results = parser.collect_all_runs()

    print(f"\nFound {len(results)} model runs")

    if not results:
        print("No results found. Check your logs folder structure.")
        return

    viz = Visualizer(results, plot_path="imgs")
    viz.plot_groups_comparison()


if __name__ == "__main__":
    main()
