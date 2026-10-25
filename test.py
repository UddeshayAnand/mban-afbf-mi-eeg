"""Reload a trained run and evaluate its deterministic notebook test split."""
import argparse
import json
from pathlib import Path
from model import build_model
from preprocessing import prepare_data
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", choices=["final", "best"], default="final")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--show-plots", action="store_true", help="Display each saved plot interactively")
    args = parser.parse_args()
    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text())
    model = build_model(config["dataset"], config["variant"])
    model.load_weights(str(run / f"{args.weights}.weights.h5"))
    splits = prepare_data(args.data_dir, config["subject"], config["dataset"])
    evaluate_model(model, {"test": splits["test"]}, run / f"test_{args.weights}",
                   config["dataset"], not args.no_tsne, args.show_plots)

if __name__ == "__main__":
    main()
