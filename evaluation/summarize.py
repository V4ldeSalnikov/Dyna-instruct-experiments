import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from euroeval.data_models import BenchmarkResult

plt.switch_backend("Agg")


EVALUATION_DIR = Path(__file__).parent
def load_metrics(config: dict, results_dir: Path) -> pd.DataFrame:
    rows = []
    for checkpoint, checkpoint_config in config["checkpoints"].items():
        for mode in ("task", "bpc"):
            path = results_dir / "raw" / checkpoint / mode / "euroeval_benchmark_results.jsonl"
            for result in BenchmarkResult.from_jsonl(path):
                totals = result.results["total"]
                metric_names = [
                    name
                    for name in totals
                    if not name.endswith("_se") and name != "num_failed_instances"
                ]
                for index, name in enumerate(metric_names):
                    rows.append(
                        {
                            "checkpoint": checkpoint,
                            "step": checkpoint_config["step"],
                            "dataset": result.dataset,
                            "task": result.task,
                            "split": "validation" if result.validation_split else "test",
                            "mode": mode,
                            "metric": name.removeprefix("test_"),
                            "value": totals[name],
                            "confidence_interval": totals.get(f"{name}_se"),
                            "failed_instances": totals["num_failed_instances"],
                            "primary": index == 0,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["mode", "dataset", "split", "step"])


def plot_progress(
    data: pd.DataFrame, mode: str, filename: str, results_dir: Path, title: str
) -> None:
    data = data[(data["mode"] == mode) & data["primary"]]
    datasets = sorted(data["dataset"].unique())
    columns = 3
    rows = math.ceil(len(datasets) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows), squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        subset = data[data["dataset"] == dataset]
        for split, split_data in subset.groupby("split"):
            split_data = split_data.sort_values("step")
            axis.errorbar(
                split_data["step"],
                split_data["value"],
                yerr=split_data["confidence_interval"],
                marker="o",
                capsize=3,
                label=split,
            )
        metric = ", ".join(sorted(subset["metric"].unique()))
        axis.set(title=dataset, xlabel="Training step", ylabel=metric)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(datasets) :]:
        axis.remove()
    direction = "lower is better" if mode == "bpc" else "higher is better"
    figure.suptitle(f"{title} {mode.upper()} progress ({direction})")
    figure.tight_layout()
    figure.savefig(results_dir / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EVALUATION_DIR / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    results_dir = EVALUATION_DIR / config["results_dir"]

    metrics = load_metrics(config, results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    primary = metrics[metrics["primary"]].drop(columns="primary")
    primary.to_csv(results_dir / "primary_metrics.csv", index=False)
    comparison = primary.pivot(
        index=["mode", "dataset", "task", "split", "metric"],
        columns="checkpoint",
        values="value",
    )
    comparison.to_csv(results_dir / "checkpoint_comparison.csv")

    plot_progress(metrics, "task", "task_metrics.png", results_dir, config["title"])
    plot_progress(metrics, "bpc", "bpc.png", results_dir, config["title"])


if __name__ == "__main__":
    main()
