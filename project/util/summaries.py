# summaries.py: Various summaries for visualization in Weights & Biases
#
# (C) 2019, Daniel Mouritzen

from typing import List, Optional, Sequence, Union

import gin
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


@gin.configurable(whitelist=['max_batch', 'fps'])
def video_summary(targets: tf.Tensor, predictions: tf.Tensor, max_batch: int = 5, fps: int = 10) -> wandb.Video:
    # Concatenate prediction and target vertically.
    frames = tf.concat([targets[:max_batch], predictions[:max_batch]], 2)
    # Stack batch entries horizontally.
    frames = tf.concat([frames[i] for i in range(frames.shape[0])], 2)
    # Convert to wandb format
    frames = tf.transpose(frames, [0, 3, 1, 2])  # wandb assumes order [time, channels, height, width]
    frames = tf.cast(frames * 255, tf.uint8)
    # Add 5 black frames to indicate beginning
    frames = tf.concat([tf.zeros_like(frames[:1])] * 5 + [frames], 0)
    return wandb.Video(frames, fps=fps, format="mp4")


def prediction_trajectory_summary(target: tf.Tensor, prediction: Optional[tf.Tensor], name: str) -> plt.Figure:
    name = name.title().replace('_', ' ')
    # Ensure that there is a feature dimension.
    if target.shape.ndims == 1:
        target = target[:, tf.newaxis]
    target = tf.unstack(tf.transpose(target, (1, 0)))
    lines: Sequence[Union[tf.Tensor, Sequence[tf.Tensor]]]
    labels: Sequence[Sequence[Optional[str]]]
    if prediction is not None:
        if prediction.shape.ndims == 1:
            prediction = prediction[:, tf.newaxis]
        prediction = tf.unstack(tf.transpose(prediction, (1, 0)))
        lines = list(zip(prediction, target))
        labels = [['Prediction', 'Truth']] * len(lines)
    else:
        lines = [target]
        labels = [list(map(str, range(len(target)))) if len(target) > 1 else [None]]
    titles = [f'{name} {i}' for i in range(len(lines))] if len(lines) > 1 else [name]
    return plot_summary(titles, lines, labels)


def plot_summary(titles: List[str],
                 lines: Sequence[Union[tf.Tensor, Sequence[tf.Tensor]]],
                 labels: Sequence[Sequence[Optional[str]]],
                 grid: bool = True,
                 ) -> plt.Figure:
    """
    Plot lines using matplotlib.

    Args:
      titles: List of titles for the subplots.
      lines: Nested list of tensors. Each list contains the lines of another
          subplot in the figure.
      labels: Nested list of strings. Each list contains the names for the lines
          of another subplot in the figure. Can be None for any of the sub plots.
      grid: Whether to add a grid to the plot.
    """
    fig, axes = plt.subplots(nrows=len(titles),
                             ncols=1,
                             sharex='all',
                             sharey='none',
                             squeeze=False,
                             figsize=(6, 3 * len(lines)))
    axes = axes[:, 0]
    for index, ax in enumerate(axes):
        ax.set_title(titles[index])
        for line, label in zip(lines[index], labels[index]):
            ax.plot(line, label=label)
        if any(labels[index]):
            ax.legend(frameon=False)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(AutoMinorLocator(10))
            ax.grid(grid, which='major', alpha=0.5)
            ax.grid(grid, which='minor', alpha=0.2)
    fig.tight_layout()
    return fig
