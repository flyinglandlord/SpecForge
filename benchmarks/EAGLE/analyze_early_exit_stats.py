"""
Utility functions for analyzing early exit hook statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def print_detailed_statistics(stats: Dict):
    """
    Print comprehensive statistics from early exit hook evaluation.
    
    Args:
        stats: Statistics dictionary returned by eagenerate_with_early_exit_hook
    """
    print("\n" + "=" * 80)
    print("EARLY EXIT HOOK EVALUATION REPORT")
    print("=" * 80)
    
    # Basic Info
    print(f"\n📊 Generation Statistics:")
    print(f"  • Total decoding steps: {stats['total_steps']}")
    print(f"  • Total tokens generated: {stats['total_tokens']}")
    print(f"  • Average tokens per step: {stats['total_tokens'] / stats['total_steps']:.2f}")
    
    # Acceptance Length Statistics
    print(f"\n✅ Accept Length Analysis:")
    print(f"  • Average TRUE accept length: {stats['avg_true_accept_length']:.4f} tokens")
    print(f"  • Average PREDICTED accept length: {stats['avg_predicted_accept_length']:.4f} tokens")
    print(f"  • Average difference (True - Pred): {stats['avg_length_difference']:.4f} tokens")
    
    if stats['avg_length_difference'] > 0:
        print(f"    → Hook UNDERESTIMATES by {abs(stats['avg_length_difference']):.4f} tokens on average")
    elif stats['avg_length_difference'] < 0:
        print(f"    → Hook OVERESTIMATES by {abs(stats['avg_length_difference']):.4f} tokens on average")
    else:
        print(f"    → Hook predictions are PERFECT on average!")
    
    # Acceptance Rate Statistics
    print(f"\n📈 Acceptance Rate Analysis:")
    print(f"  • True acceptance rate: {stats['true_acceptance_rate']:.4f} ({stats['true_acceptance_rate']*100:.2f}%)")
    print(f"  • Predicted acceptance rate: {stats['predicted_acceptance_rate']:.4f} ({stats['predicted_acceptance_rate']*100:.2f}%)")
    print(f"  • Acceptance rate gap: {stats['acceptance_rate_gap']:.4f} ({stats['acceptance_rate_gap']*100:.2f}%)")
    print(f"  • Relative error: {(stats['acceptance_rate_gap'] / stats['true_acceptance_rate'] * 100):.2f}%" if stats['true_acceptance_rate'] > 0 else "N/A")
    
    # Early Exit Statistics
    print(f"\n🚪 Early Exit Behavior:")
    print(f"  • Early exit rate: {stats['early_exit_rate']:.4f} ({stats['early_exit_rate']*100:.2f}%)")
    print(f"    (Percentage of steps where hook predicted < true accept length)")
    
    # Distribution Analysis
    if stats['true_accept_lengths']:
        true_lengths = np.array(stats['true_accept_lengths'])
        pred_lengths = np.array(stats['predicted_accept_lengths'])
        
        print(f"\n📉 Distribution Statistics:")
        print(f"  True Accept Lengths:")
        print(f"    • Min: {true_lengths.min()}")
        print(f"    • Max: {true_lengths.max()}")
        print(f"    • Median: {np.median(true_lengths):.2f}")
        print(f"    • Std Dev: {np.std(true_lengths):.2f}")
        
        print(f"  Predicted Accept Lengths:")
        print(f"    • Min: {pred_lengths.min()}")
        print(f"    • Max: {pred_lengths.max()}")
        print(f"    • Median: {np.median(pred_lengths):.2f}")
        print(f"    • Std Dev: {np.std(pred_lengths):.2f}")
        
        # Correlation
        if len(true_lengths) > 1:
            correlation = np.corrcoef(true_lengths, pred_lengths)[0, 1]
            print(f"\n🔗 Correlation:")
            print(f"  • Pearson correlation coefficient: {correlation:.4f}")
            
            if correlation > 0.8:
                print(f"    → Strong positive correlation - Hook predictions are reliable!")
            elif correlation > 0.5:
                print(f"    → Moderate correlation - Hook captures some trends")
            else:
                print(f"    → Weak correlation - Hook needs improvement")
        
        # Accuracy metrics
        exact_matches = np.sum(true_lengths == pred_lengths)
        within_1 = np.sum(np.abs(true_lengths - pred_lengths) <= 1)
        within_2 = np.sum(np.abs(true_lengths - pred_lengths) <= 2)
        
        print(f"\n🎯 Prediction Accuracy:")
        print(f"  • Exact matches: {exact_matches}/{len(true_lengths)} ({exact_matches/len(true_lengths)*100:.2f}%)")
        print(f"  • Within ±1 token: {within_1}/{len(true_lengths)} ({within_1/len(true_lengths)*100:.2f}%)")
        print(f"  • Within ±2 tokens: {within_2}/{len(true_lengths)} ({within_2/len(true_lengths)*100:.2f}%)")
    
    print("\n" + "=" * 80 + "\n")


def plot_accept_length_comparison(stats: Dict, save_path: str = None):
    """
    Create visualizations comparing true and predicted accept lengths.
    
    Args:
        stats: Statistics dictionary
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    true_lengths = stats['true_accept_lengths']
    pred_lengths = stats['predicted_accept_lengths']
    steps = list(range(1, len(true_lengths) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    axes[0, 0].plot(steps, true_lengths, 'b-', label='True Accept Length', linewidth=2)
    axes[0, 0].plot(steps, pred_lengths, 'r--', label='Predicted Accept Length', linewidth=2)
    axes[0, 0].set_xlabel('Decoding Step')
    axes[0, 0].set_ylabel('Accept Length')
    axes[0, 0].set_title('Accept Length Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(true_lengths, pred_lengths, alpha=0.6)
    max_val = max(max(true_lengths), max(pred_lengths))
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    axes[0, 1].set_xlabel('True Accept Length')
    axes[0, 1].set_ylabel('Predicted Accept Length')
    axes[0, 1].set_title('Prediction vs Truth')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution histograms
    axes[1, 0].hist(true_lengths, alpha=0.5, label='True', bins=20, color='blue')
    axes[1, 0].hist(pred_lengths, alpha=0.5, label='Predicted', bins=20, color='red')
    axes[1, 0].set_xlabel('Accept Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    errors = np.array(true_lengths) - np.array(pred_lengths)
    axes[1, 1].hist(errors, bins=20, color='green', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].set_xlabel('Error (True - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Prediction Error Distribution\nMean Error: {np.mean(errors):.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def export_stats_to_csv(stats: Dict, filename: str):
    """
    Export detailed statistics to CSV for further analysis.
    
    Args:
        stats: Statistics dictionary
        filename: Output CSV filename
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Step', 'True_Accept_Length', 'Predicted_Accept_Length', 
                        'Difference', 'Early_Exit_Triggered'])
        
        # Write data
        for i in range(len(stats['true_accept_lengths'])):
            writer.writerow([
                i + 1,
                stats['true_accept_lengths'][i],
                stats['predicted_accept_lengths'][i],
                stats['true_accept_lengths'][i] - stats['predicted_accept_lengths'][i],
                stats['early_exit_triggered'][i]
            ])
    
    print(f"Detailed statistics exported to {filename}")


def compare_multiple_hooks(results: Dict[str, Dict]):
    """
    Compare statistics from multiple different hooks.
    
    Args:
        results: Dictionary mapping hook names to their statistics
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF MULTIPLE HOOKS")
    print("=" * 80)
    
    print(f"\n{'Hook Name':<25} {'Avg True':<12} {'Avg Pred':<12} {'Gap':<10} {'Early Exit %':<15}")
    print("-" * 80)
    
    for hook_name, stats in results.items():
        print(f"{hook_name:<25} "
              f"{stats['avg_true_accept_length']:<12.4f} "
              f"{stats['avg_predicted_accept_length']:<12.4f} "
              f"{stats['acceptance_rate_gap']:<10.4f} "
              f"{stats['early_exit_rate']*100:<15.2f}")
    
    print("=" * 80 + "\n")
    
    # Find best hook
    best_hook = min(results.items(), key=lambda x: x[1]['acceptance_rate_gap'])
    print(f"🏆 Best Hook (lowest acceptance rate gap): {best_hook[0]}")
    print(f"   Gap: {best_hook[1]['acceptance_rate_gap']:.4f}")


if __name__ == "__main__":
    # Example usage
    print("This module provides utilities for analyzing early exit hook statistics.")
    print("\nExample usage:")
    print("""
    from analyze_early_exit_stats import print_detailed_statistics, plot_accept_length_comparison
    
    # After running evaluation
    output_ids, stats = model.eagenerate_with_early_exit_hook(
        input_ids=input_ids,
        log=True,
        early_exit_hook=your_hook,
    )
    
    # Analyze results
    print_detailed_statistics(stats)
    plot_accept_length_comparison(stats, save_path='results.png')
    export_stats_to_csv(stats, 'results.csv')
    """)
