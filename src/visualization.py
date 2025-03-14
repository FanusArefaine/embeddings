# src/visualization.py

import matplotlib.pyplot as plt

def print_metrics_table(metrics_dict):
    """
    Prints each metric in a clean table-like format.
    """
    print("Metric              Value")
    print("-------------------------")
    for metric_name, value in metrics_dict.items():
        print(f"{metric_name:<20} {value:.4f}")


def plot_metrics_bar_chart(metrics_dict):
    """
    Creates a simple bar chart comparing metric values.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.bar(names, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)  # Most retrieval metrics range between 0 and 1
    plt.title("Retrieval Metrics")
    plt.tight_layout()
    plt.show()
