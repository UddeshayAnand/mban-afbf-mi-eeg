"""Notebook metrics and plots, saved for each split without interactive windows."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, classification_report, confusion_matrix)
from plots import (plot_confusion_matrix, plot_confusion_matrix_percent,
                   plot_and_save_tsne)
from preprocessing import model_inputs

def evaluate_model(model, splits, output, dataset="2a", tsne=True, show_plots=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ["Left hand", "Right hand", "Foot", "Tongue"][:4 if dataset == "2a" else 2]
    labels = list(range(len(names)))
    metrics = {}
    for split, (data, truth) in splits.items():
        prob = model.predict(model_inputs(model, data), verbose=0)
        pred = prob.argmax(axis=1)
        metrics[split] = dict(accuracy=accuracy_score(truth, pred),
            precision=precision_score(truth, pred, average="weighted", zero_division=0),
            recall=recall_score(truth, pred, average="weighted", zero_division=0),
            f1=f1_score(truth, pred, average="weighted", zero_division=0),
            kappa=cohen_kappa_score(truth, pred))
        np.savez_compressed(output / f"{split}_predictions.npz", y_true=truth, y_pred=pred, probabilities=prob)
        (output / f"{split}_report.txt").write_text(classification_report(
            truth, pred, labels=labels, target_names=names, digits=4, zero_division=0))
        cm = confusion_matrix(truth, pred, labels=labels)
        np.save(output / f"{split}_confusion_matrix.npy", cm)
        plot_confusion_matrix(cm, names, output / f"{split}_confusion_matrix.png", show_plots)
        plot_confusion_matrix_percent(cm, names, f"{split} Confusion Matrix (Percentages)",
            output / f"{split}_confusion_matrix_percent.png", show_plots)
        if tsne:
            plot_and_save_tsne(prob, truth, split, output, names, dataset, show_plots)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
