
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os
import json
def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Loads metrics data from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A dictionary containing the metrics data.
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{filepath}'.")
        return {}

def plot_confusion_matrix(matrix: List[List[int]], class_names: List[str], filename: str = "confusion_matrix.png"):
    """
    Generates and saves a confusion matrix plot.

    Args:
        matrix: A 2D list representing the confusion matrix.
        filename: The name of the file to save the plot to.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout() 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_per_class_metric(scores: List[float], metric_name: str, color: str, filename: str, class_names: List[str]):
    """
    Generates and saves a bar plot for a per-class metric with value labels.

    Args:
        scores: A list of scores for each class.
        metric_name: The name of the metric (e.g., 'F1-score', 'Precision').
        color: The color for the bars in the plot.
        filename: The name of the file to save the plot to.
    """
    plt.figure(figsize=(12, 7)) # Increased figure size for better label visibility
    num_classes = len(scores)
    bars = plt.bar(range(num_classes), scores, color=color)
    
    # Add the value label on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', 
                 ha='center', va='bottom')

    plt.xlabel('Class')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Class')
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")

    plt.ylim(0, 1.1) # Adjust y-limit to make space for labels
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout() 
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def print_scalar_metrics(metrics: Dict[str, Any]):
    """
    Prints the scalar performance metrics in a formatted way.

    Args:
        metrics: A dictionary containing the metrics data.
    """
    print("\n--- Scalar Metrics ---")
    print(f"Accuracy:        {metrics.get('accuracy', 0):.4f}")
    print(f"Top-5 Accuracy:  {metrics.get('accuracy_top5', 0):.4f}")
    print(f"F1 Macro:        {metrics.get('f1_macro', 0):.4f}")
    print(f"Precision Macro: {metrics.get('precision_macro', 0):.4f}")
    print(f"Recall Macro:    {metrics.get('recall_macro', 0):.4f}")
    print("----------------------\n")


def main(file, metric_type, class_names):
    metrics_data = load_metrics(file)
    if not metrics_data:
        return # Stop execution if data loading failed

    # basename is the dir of  the file
    basename = os.path.dirname(file)
    # create a folder for the metric type
    os.makedirs(os.path.join(basename, metric_type), exist_ok=True)
    # make it the new basename
    basename = os.path.join(basename, metric_type)

    # filename is  the basename plus the start of the filename (everything before the first _)
    filename_start = metric_type

    print_scalar_metrics(metrics_data)

    plot_confusion_matrix(metrics_data.get('confusion_matrix', []), class_names=class_names, filename=os.path.join(basename, f"{filename_start}_confusion_matrix.png"))

    plot_per_class_metric(
            scores=metrics_data.get('f1_per_class', []),
            metric_name='F1-score',
            color='skyblue',
            filename=os.path.join(basename, f"{filename_start}_f1_per_class_with_labels.png"),
            class_names=class_names
        )

    plot_per_class_metric(
            scores=metrics_data.get('precision_per_class', []),
            metric_name='Precision',
            color='lightgreen',
            filename=os.path.join(basename, f"{filename_start}_precision_per_class_with_labels.png"),
            class_names=class_names
        )

    plot_per_class_metric(
            scores=metrics_data.get('recall_per_class', []),
            metric_name='Recall',
            color='salmon',
            filename=os.path.join(basename, f"{filename_start}_recall_per_class_with_labels.png"),
            class_names=class_names
        )
