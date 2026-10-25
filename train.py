"""Train the notebook models independently for each requested subject."""
import argparse
import json
import random
import os
from pathlib import Path
import time
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from model import build_model, BUILDERS
from preprocessing import prepare_data, model_inputs
from evaluate import evaluate_model
from plots import plot_history

class ThresholdStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.0254):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get("val_loss")
        if current is not None and current <= self.threshold:
            self.model.stop_training = True

def train_subject(data_dir, output_dir, subject, dataset="2a", variant="full",
                  epochs=None, batch_size=64, tsne=True, show_plots=False):
    if dataset == "2b" and variant != "full":
        raise ValueError("Only the full model is supplied for 2b")
    epochs = epochs if epochs is not None else (1200 if dataset == "2b" else 1320)
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    run = Path(output_dir) / dataset / variant / f"subject_{subject:02d}"
    run.mkdir(parents=True, exist_ok=False)
    tf.keras.backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    config = dict(subject=subject, dataset=dataset, variant=variant, epochs=epochs,
                  batch_size=batch_size, seed=42, split="notebook-pooled-windows",
                  tensorflow=tf.__version__, weights="final.weights.h5")
    (run / "config.json").write_text(json.dumps(config, indent=2))
    splits = prepare_data(data_dir, subject, dataset)
    model = build_model(dataset, variant)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    with (run / "model_summary.txt").open("w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    callbacks = []
    # Full 2a had callbacks commented out; ablations and 2b enabled them.
    if dataset == "2b" or variant != "full":
        callbacks = [ThresholdStopping(0.025 if dataset == "2b" else 0.0254),
                     tf.keras.callbacks.ModelCheckpoint(str(run / "best.weights.h5"),
                        monitor="val_loss", save_best_only=True, save_weights_only=True)]
        if dataset == "2b":
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=1e-5))
    classes = 4 if dataset == "2a" else 2
    xtr, ytr = splits["train"]
    xval, yval = splits["val"]
    start = time.perf_counter()
    history = model.fit(model_inputs(model, xtr), tf.keras.utils.to_categorical(ytr, classes),
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2, callbacks=callbacks,
        validation_data=(model_inputs(model, xval), tf.keras.utils.to_categorical(yval, classes)))
    model.save_weights(str(run / "final.weights.h5"))
    np.savez(run / "history.npz", **history.history)
    plot_history(history.history, run, show_plots)
    config["training_seconds"] = time.perf_counter() - start
    config["epochs_completed"] = len(history.history["loss"])
    (run / "config.json").write_text(json.dumps(config, indent=2))
    return evaluate_model(model, splits, run, dataset, tsne, show_plots)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", choices=["2a", "2b"], default="2a")
    parser.add_argument("--subjects", nargs="+", type=int, choices=range(1,10), default=[1])
    parser.add_argument("--variant", choices=list(BUILDERS), default="full")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--show-plots", action="store_true", help="Display each saved plot interactively")
    args = parser.parse_args()
    for subject in dict.fromkeys(args.subjects):
        train_subject(args.data_dir, args.output_dir, subject, args.dataset, args.variant,
                      args.epochs, args.batch_size, not args.no_tsne, args.show_plots)

if __name__ == "__main__":
    main()
