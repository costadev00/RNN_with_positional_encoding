import argparse
import csv
import json
import math
import os
import pathlib
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from PCS5024_EP_Time_Series import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_FILENAME,
    DEFAULT_FUTURE_LEN,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PAST_LEN,
    DEFAULT_POS_ENCODING_DIM,
    DEFAULT_SLIDING_WINDOW_STEP,
    DEFAULT_TIME_SCALE,
    DEFAULT_TRAIN_TEST_SPLIT_DATE,
    DEFAULT_WEIGHT_DECAY,
    SEED,
)


@dataclass(frozen=True)
class Scenario:
    code: str
    model_variant: str
    missing_ratio: float
    objective: str

    @property
    def model_label(self) -> str:
        return "GRU base" if self.model_variant == "gru_base" else "GRU temporal"

    @property
    def slug(self) -> str:
        model_part = "GRU_base" if self.model_variant == "gru_base" else "GRU_temporal"
        missing_part = f"{self.missing_ratio:.1f}".replace(".", "p")
        return f"cenario_{self.code}_{model_part}_missing_{missing_part}"


SCENARIOS = [
    Scenario("A", "gru_base", 0.0, "baseline ideal"),
    Scenario("B", "gru_base", 0.1, "robustez"),
    Scenario("C", "gru_base", 0.3, "degradacao principal"),
    Scenario("D", "gru_base", 0.5, "degradacao severa"),
    Scenario("E", "gru_temporal", 0.0, "controle"),
    Scenario("F", "gru_temporal", 0.1, "robustez"),
    Scenario("G", "gru_temporal", 0.3, "recuperacao principal"),
    Scenario("H", "gru_temporal", 0.5, "robustez severa"),
]


def get_gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def metric_path(reports_dir: pathlib.Path, scenario: Scenario) -> pathlib.Path:
    return reports_dir / f"{scenario.slug}_metrics.json"


def history_path(reports_dir: pathlib.Path, scenario: Scenario) -> pathlib.Path:
    return reports_dir / f"{scenario.slug}_history.csv"


def predictions_path(reports_dir: pathlib.Path, scenario: Scenario) -> pathlib.Path:
    return reports_dir / f"{scenario.slug}_predictions.csv"


def required_outputs_exist(reports_dir: pathlib.Path, scenario: Scenario) -> bool:
    return all(
        path.exists()
        for path in [
            metric_path(reports_dir, scenario),
            history_path(reports_dir, scenario),
            predictions_path(reports_dir, scenario),
        ]
    )


def build_command(
    python_exe: str,
    script_path: pathlib.Path,
    reports_dir: pathlib.Path,
    scenario: Scenario,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python_exe,
        str(script_path),
        "--model_variant",
        scenario.model_variant,
        "--data_removal_ratio",
        str(scenario.missing_ratio),
        "--run_name",
        scenario.slug,
        "--objective",
        scenario.objective,
        "--output_dir",
        str(reports_dir),
        "--predictions_csv",
        f"{scenario.slug}_predictions.csv",
        "--history_csv",
        f"{scenario.slug}_history.csv",
        "--metrics_json",
        f"{scenario.slug}_metrics.json",
        "--loss_curve_png",
        f"{scenario.slug}_loss_raw.png",
        "--model_weights_path",
        f"{scenario.slug}_model_weights.pth",
        "--data_filename",
        args.data_filename,
        "--split_date",
        args.split_date,
        "--past_len",
        str(args.past_len),
        "--future_len",
        str(args.future_len),
        "--sliding_window_step",
        str(args.sliding_window_step),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--torch_threads",
        str(args.torch_threads),
        "--hidden_size",
        str(args.hidden_size),
        "--num_epochs",
        str(args.num_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--pos_encoding_dim",
        str(args.pos_encoding_dim),
        "--time_scale",
        str(args.time_scale),
        "--seed",
        str(args.seed),
    ]


def run_scenario(
    scenario: Scenario,
    gpu_id: int | None,
    args: argparse.Namespace,
    repo_root: pathlib.Path,
    reports_dir: pathlib.Path,
) -> tuple[str, str]:
    if not args.force and required_outputs_exist(reports_dir, scenario):
        return scenario.code, "skipped"

    log_path = reports_dir / f"{scenario.slug}_train.log"
    command = build_command(
        python_exe=args.python,
        script_path=repo_root / "PCS5024_EP_Time_Series.py",
        reports_dir=reports_dir,
        scenario=scenario,
        args=args,
    )
    env = os.environ.copy()
    env["DISABLE_TQDM"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with log_path.open("w") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Scenario {scenario.code} failed with exit code "
            f"{completed.returncode}. See {log_path}"
        )

    return scenario.code, "trained"


def load_metrics(reports_dir: pathlib.Path) -> list[dict[str, object]]:
    rows = []
    scenario_by_code = {scenario.code: scenario for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        with metric_path(reports_dir, scenario).open() as json_file:
            metrics = json.load(json_file)
        metrics["scenario"] = scenario.code
        metrics["model_label"] = scenario.model_label
        metrics["objective"] = scenario_by_code[scenario.code].objective
        metrics["missing_ratio"] = scenario.missing_ratio
        metrics["slug"] = scenario.slug
        rows.append(metrics)
    return rows


def write_summary_csv(rows: list[dict[str, object]], output_path: pathlib.Path) -> None:
    fieldnames = [
        "scenario",
        "model_label",
        "missing_ratio",
        "num_epochs",
        "objective",
        "final_train_loss",
        "final_test_loss",
        "best_test_loss",
        "mse",
        "rmse",
        "mae",
        "bias",
        "num_windows",
        "num_points",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(rows: list[dict[str, object]], output_path: pathlib.Path) -> None:
    lines = [
        r"\begin{tabular}{lllllrrrr}",
        r"\hline",
        r"Cenario & Modelo & Missing & Epocas & Objetivo & Loss teste & MSE & RMSE & MAE \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["scenario"]),
                    latex_escape(row["model_label"]),
                    f"{float(row['missing_ratio']):.1f}",
                    str(int(row["num_epochs"])),
                    latex_escape(row["objective"]),
                    f"{float(row['final_test_loss']):.4f}",
                    f"{float(row['mse']):.4f}",
                    f"{float(row['rmse']):.4f}",
                    f"{float(row['mae']):.4f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    output_path.write_text("\n".join(lines))


def padded_limits(values: list[float], pad_fraction: float = 0.05) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 1.0
    low = min(finite_values)
    high = max(finite_values)
    if low == high:
        pad = abs(low) * pad_fraction if low else 1.0
        return low - pad, high + pad
    pad = (high - low) * pad_fraction
    return low - pad, high + pad


def load_history(reports_dir: pathlib.Path, scenario: Scenario) -> pl.DataFrame:
    return pl.read_csv(history_path(reports_dir, scenario))


def plot_standardized_losses(reports_dir: pathlib.Path) -> None:
    histories = {scenario.code: load_history(reports_dir, scenario) for scenario in SCENARIOS}
    max_epoch = max(int(history["epoch"].max()) for history in histories.values())
    x_high = max(2, max_epoch)
    loss_values = []
    for history in histories.values():
        loss_values.extend(history["train_loss"].to_list())
        loss_values.extend(history["test_loss"].to_list())
    y_limits = padded_limits([float(value) for value in loss_values])

    for scenario in SCENARIOS:
        history = histories[scenario.code]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(history["epoch"], history["train_loss"], label="Treino", linewidth=2)
        ax.plot(history["epoch"], history["test_loss"], label="Teste", linewidth=2)
        ax.set_xlim(1, x_high)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss normalizada (MSE)")
        ax.set_title(
            f"Cenario {scenario.code}: {scenario.model_label}, "
            f"missing={scenario.missing_ratio:.1f}"
        )
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / f"{scenario.slug}_loss.png", dpi=160)
        plt.close(fig)


def selected_prediction_windows(reports_dir: pathlib.Path) -> dict[str, pl.DataFrame]:
    windows = {}
    for scenario in SCENARIOS:
        predictions = pl.read_csv(predictions_path(reports_dir, scenario))
        selected_window = int(predictions["window_id"].max()) // 2
        window_df = predictions.filter(pl.col("window_id") == selected_window)
        if window_df.is_empty():
            window_df = predictions.filter(pl.col("window_id") == 0)
        windows[scenario.code] = window_df
    return windows


def plot_standardized_forecasts(reports_dir: pathlib.Path) -> None:
    windows = selected_prediction_windows(reports_dir)
    max_step = max(int(window["step"].max()) for window in windows.values())
    y_values = []
    for window in windows.values():
        y_values.extend(window["Y"].to_list())
        y_values.extend(window["Y_hat"].to_list())
    y_limits = padded_limits([float(value) for value in y_values])

    for scenario in SCENARIOS:
        window = windows[scenario.code]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(window["step"], window["Y"], label="Observado", linewidth=2)
        ax.plot(window["step"], window["Y_hat"], label="Previsto", linewidth=2)
        ax.set_xlim(0, max_step)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Passo futuro")
        ax.set_ylabel("SSH")
        ax.set_title(
            f"Cenario {scenario.code}: {scenario.model_label}, "
            f"missing={scenario.missing_ratio:.1f}"
        )
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / f"{scenario.slug}_forecast.png", dpi=160)
        plt.close(fig)


def plot_metric_comparison(rows: list[dict[str, object]], reports_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for model_label in ["GRU base", "GRU temporal"]:
        model_rows = [row for row in rows if row["model_label"] == model_label]
        model_rows = sorted(model_rows, key=lambda row: float(row["missing_ratio"]))
        x = [float(row["missing_ratio"]) for row in model_rows]
        rmse = [float(row["rmse"]) for row in model_rows]
        mae = [float(row["mae"]) for row in model_rows]
        axes[0].plot(x, rmse, marker="o", linewidth=2, label=model_label)
        axes[1].plot(x, mae, marker="o", linewidth=2, label=model_label)

    axes[0].set_title("RMSE por taxa de missing")
    axes[0].set_ylabel("RMSE")
    axes[1].set_title("MAE por taxa de missing")
    axes[1].set_ylabel("MAE")
    for ax in axes:
        ax.set_xlabel("Missing ratio")
        ax.set_xticks([0.0, 0.1, 0.3, 0.5])
        ax.grid(True, alpha=0.35)
        ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "comparativo_rmse_mae.png", dpi=170)
    plt.close(fig)


def regenerate_reports(reports_dir: pathlib.Path) -> None:
    rows = load_metrics(reports_dir)
    write_summary_csv(rows, reports_dir / "resumo_resultados.csv")
    write_latex_table(rows, reports_dir / "tabela_resultados.tex")
    plot_standardized_losses(reports_dir)
    plot_standardized_forecasts(reports_dir)
    plot_metric_comparison(rows, reports_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the eight comparative GRU scenarios and generate reports."
    )
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=8)
    parser.add_argument("--data_filename", default=DEFAULT_DATA_FILENAME)
    parser.add_argument("--split_date", default=DEFAULT_TRAIN_TEST_SPLIT_DATE)
    parser.add_argument("--past_len", type=int, default=DEFAULT_PAST_LEN)
    parser.add_argument("--future_len", type=int, default=DEFAULT_FUTURE_LEN)
    parser.add_argument("--sliding_window_step", type=int, default=DEFAULT_SLIDING_WINDOW_STEP)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--pos_encoding_dim", type=int, default=DEFAULT_POS_ENCODING_DIM)
    parser.add_argument("--time_scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    reports_dir = (repo_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        gpu_count = 0 if args.cpu else get_gpu_count()
        default_parallel = min(4, gpu_count) if gpu_count > 0 else 1
        max_parallel = args.max_parallel if args.max_parallel > 0 else default_parallel
        max_parallel = max(1, max_parallel)

        print(
            f"Running {len(SCENARIOS)} scenarios with max_parallel={max_parallel} "
            f"and gpu_count={gpu_count}."
        )

        gpu_queue: queue.Queue[int] | None = None
        if gpu_count > 0:
            gpu_queue = queue.Queue()
            for gpu_id in range(gpu_count):
                gpu_queue.put(gpu_id)

        def submit_scenario(scenario: Scenario) -> tuple[str, str]:
            gpu_id = None
            if gpu_queue is not None:
                gpu_id = gpu_queue.get()
            try:
                return run_scenario(scenario, gpu_id, args, repo_root, reports_dir)
            finally:
                if gpu_queue is not None and gpu_id is not None:
                    gpu_queue.put(gpu_id)

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = []
            for scenario in SCENARIOS:
                futures.append(executor.submit(submit_scenario, scenario))
            for future in as_completed(futures):
                scenario_code, status = future.result()
                print(f"Scenario {scenario_code}: {status}")

    regenerate_reports(reports_dir)
    print(f"Reports generated in {reports_dir}")
    print(f"LaTeX table: {reports_dir / 'tabela_resultados.tex'}")


if __name__ == "__main__":
    main()
