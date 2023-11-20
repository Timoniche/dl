import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_metrics(metrics_path):
    df = pd.read_csv(metrics_path)
    metric_names = [
        'training_f1_macro_epoch_mean',
        'training_cross_entropy_epoch_mean',
    ]
    for metric_name in metric_names:
        plot_metric(metrics_csv=df, metric_name=metric_name)


def plot_metric(
        metrics_csv,
        metric_name,
):
    metrics = list(metrics_csv[metric_name])[:-1]
    training_samples = len(metrics)
    plt.plot(
        [*range(training_samples)],
        metrics
    )
    plt.xticks(np.arange(training_samples))
    plt.title('Training metrics')
    plt.xlabel('epoch')
    plt.ylabel(metric_name.removeprefix('training_'))
    plt.show()
