import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd

class BeamWidthHeatmapVisualizer:
    """
    A class to create heatmap visualizations comparing WER and CER across different beam widths.
    
    This class loads beam width comparison results and generates heatmaps showing the performance
    metrics (WER and CER) for different beam width configurations.
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the visualizer with a JSON file containing beam width comparison results.
        
        Args:
            json_file_path: Path to the JSON file containing the results
        """
        self.data = self._load_data(json_file_path)
        self.beam_widths = []
        self.wer_scores = []
        self.cer_scores = []
        self.inference_times = []
        
        self._extract_metrics()
        
    def _load_data(self, json_file_path: str) -> Dict:
        """Load and parse the JSON data."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_metrics(self):
        """Extract metrics from the loaded data."""
        for result in self.data['results']:
            beam_width = result['params']['beam_width']
            wer = result['eval']['beam']['metrics']['wer']
            cer = result['eval']['beam']['metrics']['cer']
            inference_time = result['eval']['beam']['avg_inference_time']
            
            self.beam_widths.append(beam_width)
            self.wer_scores.append(wer)
            self.cer_scores.append(cer)
            self.inference_times.append(inference_time)
    
    def create_full_analysis(self, save_path: Optional[str] = None):
        """
        Create a complete analysis with multiple heatmaps and metrics table.
        
        Args:
            save_path: Optional path to save the figure. If None, just displays.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. WER Heatmap
        wer_data = np.array(self.wer_scores).reshape(-1, 1) * 100
        sns.heatmap(wer_data, 
                   annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=['WER'], yticklabels=self.beam_widths,
                   cbar_kws={'label': 'WER'}, ax=axes[0])
        axes[0].set_title('Word Error Rate (WER), %', fontsize=12)
        axes[0].set_ylabel('Beam Width')
        
        # 2. CER Heatmap
        cer_data = np.array(self.cer_scores).reshape(-1, 1) * 100
        sns.heatmap(cer_data, 
                   annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=['CER'], yticklabels=self.beam_widths,
                   cbar_kws={'label': 'CER'}, ax=axes[1])
        axes[1].set_title('Character Error Rate (CER), %', fontsize=12)
        axes[1].set_ylabel('Beam Width')
        
        # 3. Inference Time Heatmap
        time_data = np.array(self.inference_times).reshape(-1, 1)
        sns.heatmap(time_data, 
                   annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=['Time (s)'], yticklabels=self.beam_widths,
                   cbar_kws={'label': 'Time (s)'}, ax=axes[2])
        axes[2].set_title('Average Inference Time', fontsize=12)
        axes[2].set_ylabel('Beam Width')
        
        # Add overall title
        fig.suptitle('Beam Width Analysis: WER, CER, and Inference Time', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get a summary DataFrame with all metrics.
        
        Returns:
            Pandas DataFrame with summary statistics
        """
        df = pd.DataFrame({
            'Beam Width': self.beam_widths,
            'WER': self.wer_scores,
            'CER': self.cer_scores,
            'Inference Time (s)': self.inference_times
        })
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = BeamWidthHeatmapVisualizer('./results/beam_width_comparison.json')
    
    # Get summary statistics
    print("Summary Statistics:")
    print(visualizer.get_summary_stats())
    print("\n")
    
    # Create full analysis with all plots
    fig_full = visualizer.create_full_analysis(save_path='./imgs/beam_width_analysis.png')
    plt.show()