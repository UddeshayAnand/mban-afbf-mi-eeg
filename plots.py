"""Plotting from the original notebooks, shared by training and evaluation."""
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE


def _save(fig, path, show=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


def plot_history(history, output, show=False):
    """Original two-panel train/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history[metric])
        ax.plot(history["val_" + metric])
        ax.set(title=metric.capitalize(), xlabel="Epoch", ylabel=metric.capitalize())
        ax.legend(["Train", "Val"], loc="upper left")
    _save(fig, Path(output) / "learning_curves.png", show)


def plot_confusion_matrix(cm, class_labels, save_path, show=False):
    size = 4 if len(class_labels) == 2 else 5
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(cmap=plt.cm.Blues, ax=ax)
    _save(fig, save_path, show)


def plot_confusion_matrix_percent(cm, class_labels, title, save_path, show=False):
    """Notebook row normalization, zero-row protection and percent annotations."""
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    percent = cm / row_sums * 100
    size = 4 if len(class_labels) == 2 else 5
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(percent, interpolation="nearest", cmap=plt.cm.Greens)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_labels)), yticks=np.arange(len(class_labels)),
           xticklabels=class_labels, yticklabels=class_labels,
           xlabel="Predicted label", ylabel="True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = percent.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{percent[i, j]:.1f}%", ha="center", va="center",
                    color="white" if percent[i, j] > threshold else "black")
    # IV-2a commented out its title; IV-2b displays it.
    if len(class_labels) == 2:
        ax.set_title(title)
    _save(fig, save_path, show)


def plot_and_save_tsne(probabilities, labels, split_name, output, class_labels,
                       dataset="2a", show=False):
    """Use softmax probabilities as in the notebooks, not hidden features.

    IV-2a has a numeric-class overview and a named-class figure sharing one
    embedding. IV-2b has only the named-class figure. The previously unsaved
    IV-2a overview is now saved too.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = probabilities.reshape(probabilities.shape[0], -1)
    if len(features) <= 30:
        warnings.warn(f"Skipping {split_name} t-SNE: notebook perplexity 30 requires "
                      "at least 31 samples.", UserWarning, stacklevel=2)
        return None
    embedding = TSNE(n_components=2, random_state=42).fit_transform(features)
    np.save(output / f"{split_name}_tsne.npy", embedding)
    unique = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    if dataset == "2a":
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls in enumerate(unique):
            idx = labels == cls
            ax.scatter(*embedding[idx].T, color=colors[i], label=f"Class {cls}", s=40)
        ax.set(title="t-SNE of output probabilities", xlabel="t-SNE Feature 1",
               ylabel="t-SNE Feature 2")
        ax.legend(title="Classes")
        ax.grid(True)
        _save(fig, output / f"{split_name}_tsne_classes.png", show)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(unique):
        idx = labels == cls
        ax.scatter(*embedding[idx].T, color=colors[i], label=class_labels[int(cls)], s=40)
    ax.set(xlabel="t-SNE Feature 1", ylabel="t-SNE Feature 2")
    if dataset == "2b":
        ax.set_title(f"t-SNE - {split_name}")
    ax.legend(title="Classes", loc="upper left")
    _save(fig, output / f"{split_name}_tsne.png", show)
    return embedding
