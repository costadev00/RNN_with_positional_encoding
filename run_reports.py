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
    DEFAULT_IQN_EVAL_SAMPLES,
    DEFAULT_IQN_QUANTILES,
    DEFAULT_IQN_TAU_EMBEDDING_DIM,
    DEFAULT_IQN_TRAIN_SAMPLES,
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


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


@dataclass(frozen=True)
class Scenario:
    code: str
    model_variant: str
    missing_ratio: float
    objective: str

    @property
    def model_label(self) -> str:
        labels = {
            "gru_base": "GRU base",
            "gru_temporal": "GRU temporal",
            "gru_temporal_iqn": "GRU temporal IQN",
        }
        return labels[self.model_variant]

    @property
    def slug(self) -> str:
        model_parts = {
            "gru_base": "GRU_base",
            "gru_temporal": "GRU_temporal",
            "gru_temporal_iqn": "GRU_temporal_IQN",
        }
        model_part = model_parts[self.model_variant]
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
    Scenario("I", "gru_temporal_iqn", 0.0, "IQN controle"),
    Scenario("J", "gru_temporal_iqn", 0.1, "IQN robustez"),
    Scenario("K", "gru_temporal_iqn", 0.3, "IQN recuperacao principal"),
    Scenario("L", "gru_temporal_iqn", 0.5, "IQN robustez severa"),
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
    training_command = [
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
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--early_stopping_min_delta",
        str(args.early_stopping_min_delta),
        "--pos_encoding_dim",
        str(args.pos_encoding_dim),
        "--time_scale",
        str(args.time_scale),
        "--iqn_train_samples",
        str(args.iqn_train_samples),
        "--iqn_eval_samples",
        str(args.iqn_eval_samples),
        "--iqn_tau_embedding_dim",
        str(args.iqn_tau_embedding_dim),
        "--iqn_quantiles",
        args.iqn_quantiles,
        "--seed",
        str(args.seed),
    ]

    if args.ddp:
        return [
            python_exe,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(args.nproc_per_node),
            *training_command,
        ]

    return [python_exe, *training_command]


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
    elif gpu_id is not None and not args.ddp:
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
        path = metric_path(reports_dir, scenario)
        if not path.exists():
            continue
        with path.open() as json_file:
            metrics = json.load(json_file)
        metrics["scenario"] = scenario.code
        metrics["model_label"] = scenario.model_label
        metrics["objective"] = scenario_by_code[scenario.code].objective
        metrics["missing_ratio"] = scenario.missing_ratio
        metrics["slug"] = scenario.slug
        rows.append(metrics)
    if not rows:
        raise FileNotFoundError(f"No scenario metrics found in {reports_dir}.")
    return rows


def write_summary_csv(rows: list[dict[str, object]], output_path: pathlib.Path) -> None:
    fieldnames = [
        "scenario",
        "model_label",
        "missing_ratio",
        "num_epochs",
        "epochs_ran",
        "best_epoch",
        "selected_model_epoch",
        "early_stopped",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "initial_weights_path",
        "objective",
        "final_train_loss",
        "final_test_loss",
        "best_test_loss",
        "mse",
        "rmse",
        "mae",
        "bias",
        "quantile_loss",
        "median_mse",
        "median_rmse",
        "median_mae",
        "coverage_80",
        "coverage_90",
        "interval_width_80",
        "interval_width_90",
        "iqn_train_samples",
        "iqn_eval_samples",
        "iqn_tau_embedding_dim",
        "iqn_quantiles",
        "num_windows",
        "num_points",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
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


def format_optional_float(row: dict[str, object], field: str, digits: int = 4) -> str:
    value = row.get(field, "")
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def write_latex_table(rows: list[dict[str, object]], output_path: pathlib.Path) -> None:
    lines = [
        r"\begin{tabular}{lllllrrrrrrr}",
        r"\hline",
        (
            r"Cenario & Modelo & Missing & Epocas & Objetivo & Loss teste & "
            r"MSE & RMSE & MAE & QLoss & Cob80 & Cob90 \\"
        ),
        r"\hline",
    ]
    for row in rows:
        epochs_value = row.get("epochs_ran", row["num_epochs"])
        lines.append(
            " & ".join(
                [
                    latex_escape(row["scenario"]),
                    latex_escape(row["model_label"]),
                    f"{float(row['missing_ratio']):.1f}",
                    str(int(epochs_value)),
                    latex_escape(row["objective"]),
                    f"{float(row['final_test_loss']):.4f}",
                    f"{float(row['mse']):.4f}",
                    f"{float(row['rmse']):.4f}",
                    f"{float(row['mae']):.4f}",
                    format_optional_float(row, "quantile_loss"),
                    format_optional_float(row, "coverage_80"),
                    format_optional_float(row, "coverage_90"),
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


def scenarios_with_history(reports_dir: pathlib.Path) -> list[Scenario]:
    return [scenario for scenario in SCENARIOS if history_path(reports_dir, scenario).exists()]


def scenarios_with_predictions(reports_dir: pathlib.Path) -> list[Scenario]:
    return [
        scenario
        for scenario in SCENARIOS
        if predictions_path(reports_dir, scenario).exists()
    ]


def plot_standardized_losses(reports_dir: pathlib.Path) -> None:
    scenarios = scenarios_with_history(reports_dir)
    if not scenarios:
        return
    histories = {scenario.code: load_history(reports_dir, scenario) for scenario in scenarios}
    max_epoch = max(int(history["epoch"].max()) for history in histories.values())
    x_high = max(2, max_epoch)
    loss_values = []
    for history in histories.values():
        loss_values.extend(history["train_loss"].to_list())
        loss_values.extend(history["test_loss"].to_list())
    y_limits = padded_limits([float(value) for value in loss_values])

    for scenario in scenarios:
        history = histories[scenario.code]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(history["epoch"], history["train_loss"], label="Treino", linewidth=2)
        ax.plot(history["epoch"], history["test_loss"], label="Teste", linewidth=2)
        ax.set_xlim(1, x_high)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss normalizada")
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
    for scenario in scenarios_with_predictions(reports_dir):
        predictions = pl.read_csv(predictions_path(reports_dir, scenario))
        selected_window = int(predictions["window_id"].max()) // 2
        window_df = predictions.filter(pl.col("window_id") == selected_window)
        if window_df.is_empty():
            window_df = predictions.filter(pl.col("window_id") == 0)
        windows[scenario.code] = window_df
    return windows


def plot_standardized_forecasts(reports_dir: pathlib.Path) -> None:
    windows = selected_prediction_windows(reports_dir)
    if not windows:
        return
    max_step = max(int(window["step"].max()) for window in windows.values())
    y_values = []
    for window in windows.values():
        y_values.extend(window["Y"].to_list())
        y_values.extend(window["Y_hat"].to_list())
        for quantile_column in ["q05", "q10", "q50", "q90", "q95"]:
            if quantile_column in window.columns:
                y_values.extend(window[quantile_column].to_list())
    y_limits = padded_limits([float(value) for value in y_values])

    for scenario in scenarios_with_predictions(reports_dir):
        window = windows[scenario.code]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        if {"q05", "q10", "q90", "q95"}.issubset(set(window.columns)):
            ax.fill_between(
                window["step"].to_numpy(),
                window["q05"].to_numpy(),
                window["q95"].to_numpy(),
                alpha=0.18,
                label="q05-q95",
            )
            ax.fill_between(
                window["step"].to_numpy(),
                window["q10"].to_numpy(),
                window["q90"].to_numpy(),
                alpha=0.28,
                label="q10-q90",
            )
        ax.plot(window["step"], window["Y"], label="Observado", linewidth=2)
        prediction_label = "q50" if "q50" in window.columns else "Previsto"
        ax.plot(window["step"], window["Y_hat"], label=prediction_label, linewidth=2)
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
    for model_label in ["GRU base", "GRU temporal", "GRU temporal IQN"]:
        model_rows = [row for row in rows if row["model_label"] == model_label]
        if not model_rows:
            continue
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


def plot_iqn_coverage(rows: list[dict[str, object]], reports_dir: pathlib.Path) -> None:
    iqn_rows = [
        row
        for row in rows
        if row.get("coverage_80") not in ("", None)
        and row.get("coverage_90") not in ("", None)
    ]
    if not iqn_rows:
        return

    iqn_rows = sorted(iqn_rows, key=lambda row: float(row["missing_ratio"]))
    x = [float(row["missing_ratio"]) for row in iqn_rows]
    coverage_80 = [float(row["coverage_80"]) for row in iqn_rows]
    coverage_90 = [float(row["coverage_90"]) for row in iqn_rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, coverage_80, marker="o", linewidth=2, label="Empirica q10-q90")
    ax.plot(x, coverage_90, marker="o", linewidth=2, label="Empirica q05-q95")
    ax.axhline(0.80, color="0.35", linestyle="--", linewidth=1.5, label="Nominal 80%")
    ax.axhline(0.90, color="0.55", linestyle=":", linewidth=1.8, label="Nominal 90%")
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("Cobertura")
    ax.set_title("Cobertura empirica da IQN")
    ax.set_xticks([0.0, 0.1, 0.3, 0.5])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "cobertura_iqn.png", dpi=170)
    plt.close(fig)


def plot_iqn_interval_width(rows: list[dict[str, object]], reports_dir: pathlib.Path) -> None:
    iqn_rows = [
        row
        for row in rows
        if row.get("interval_width_80") not in ("", None)
        and row.get("interval_width_90") not in ("", None)
    ]
    if not iqn_rows:
        return

    iqn_rows = sorted(iqn_rows, key=lambda row: float(row["missing_ratio"]))
    x = [float(row["missing_ratio"]) for row in iqn_rows]
    width_80 = [float(row["interval_width_80"]) for row in iqn_rows]
    width_90 = [float(row["interval_width_90"]) for row in iqn_rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, width_80, marker="o", linewidth=2, label="Largura q10-q90")
    ax.plot(x, width_90, marker="o", linewidth=2, label="Largura q05-q95")
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("Largura media")
    ax.set_title("Largura media dos intervalos IQN")
    ax.set_xticks([0.0, 0.1, 0.3, 0.5])
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "largura_intervalos_iqn.png", dpi=170)
    plt.close(fig)


def regenerate_reports(reports_dir: pathlib.Path) -> None:
    rows = load_metrics(reports_dir)
    write_summary_csv(rows, reports_dir / "resumo_resultados.csv")
    write_latex_table(rows, reports_dir / "tabela_resultados.tex")
    plot_standardized_losses(reports_dir)
    plot_standardized_forecasts(reports_dir)
    plot_metric_comparison(rows, reports_dir)
    plot_iqn_coverage(rows, reports_dir)
    plot_iqn_interval_width(rows, reports_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final PCS5024 experiment: scenarios A-L comparing GRU base, "
            "GRU temporal, and GRU temporal IQN."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  Full 4-GPU run:\n"
            "    python run_reports.py --ddp --effective_batch_size 1024 "
            "--early_stopping_patience 50 --early_stopping_min_delta 0.0\n\n"
            "  Regenerate tables and plots from existing lightweight artifacts:\n"
            "    python run_reports.py --skip_training\n\n"
            "  CPU/debug run:\n"
            "    python run_reports.py --cpu --num_epochs 1 --hidden_size 16 "
            "--batch_size 32 --iqn_train_samples 2 --iqn_eval_samples 8"
        ),
    )
    parser.add_argument(
        "--reports_dir",
        default="reports_iqn_temporal",
        help="Directory used for final metrics, histories, figures, and tables.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch scenario training.")
    parser.add_argument("--force", action="store_true", help="Retrain scenarios even if full outputs already exist.")
    parser.add_argument("--skip_training", action="store_true", help="Only regenerate summary tables and plots from existing artifacts.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA and run on CPU.")
    parser.add_argument("--ddp", action="store_true", help="Use torch DistributedDataParallel for each scenario.")
    parser.add_argument(
        "--nproc_per_node",
        "--nproc-per-node",
        type=int,
        default=0,
        help="Number of DDP processes. Use 0 to auto-detect all CUDA GPUs.",
    )
    parser.add_argument(
        "--effective_batch_size",
        "--effective-batch-size",
        type=int,
        default=0,
        help="Desired global batch size for DDP. When set, per-process batch size is derived automatically.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=0,
        help="How many non-DDP scenarios to run concurrently. Use 0 for an automatic value.",
    )
    parser.add_argument("--torch_threads", type=int, default=8, help="CPU threads passed to the training script.")
    parser.add_argument("--data_filename", default=DEFAULT_DATA_FILENAME, help="Input SSH CSV file.")
    parser.add_argument("--split_date", default=DEFAULT_TRAIN_TEST_SPLIT_DATE, help="Train/test split timestamp.")
    parser.add_argument("--past_len", type=int, default=DEFAULT_PAST_LEN, help="Past context horizon, in timestamp units.")
    parser.add_argument("--future_len", type=int, default=DEFAULT_FUTURE_LEN, help="Forecast horizon, in timestamp units.")
    parser.add_argument(
        "--sliding_window_step",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_STEP,
        help="Sliding-window stride. Smaller values create more training windows.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per process.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE, help="GRU hidden size.")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Maximum training epochs.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Epochs without test-loss improvement before stopping. Use 0 to disable.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, help="Minimum test-loss reduction required to count as an improvement.")
    parser.add_argument("--pos_encoding_dim", type=int, default=DEFAULT_POS_ENCODING_DIM, help="Sinusoidal temporal encoding dimension.")
    parser.add_argument("--time_scale", type=float, default=DEFAULT_TIME_SCALE, help="Scale applied to relative timestamps before temporal encoding.")
    parser.add_argument("--iqn_train_samples", type=int, default=DEFAULT_IQN_TRAIN_SAMPLES, help="Tau samples per valid target point during IQN training.")
    parser.add_argument("--iqn_eval_samples", type=int, default=DEFAULT_IQN_EVAL_SAMPLES, help="Deterministic tau samples per point during IQN evaluation.")
    parser.add_argument(
        "--iqn_tau_embedding_dim",
        type=int,
        default=DEFAULT_IQN_TAU_EMBEDDING_DIM,
        help="Cosine basis size for the IQN tau embedding.",
    )
    parser.add_argument("--iqn_quantiles", default=DEFAULT_IQN_QUANTILES, help="Comma-separated quantiles exported for IQN scenarios.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for all scenarios.")
    args = parser.parse_args()
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience must be >= 0.")
    if args.early_stopping_min_delta < 0.0:
        parser.error("--early_stopping_min_delta must be >= 0.0.")
    if args.nproc_per_node < 0:
        parser.error("--nproc_per_node must be >= 0.")
    if args.effective_batch_size < 0:
        parser.error("--effective_batch_size must be >= 0.")
    if args.iqn_train_samples <= 0:
        parser.error("--iqn_train_samples must be > 0.")
    if args.iqn_eval_samples <= 0:
        parser.error("--iqn_eval_samples must be > 0.")
    if args.iqn_tau_embedding_dim <= 0:
        parser.error("--iqn_tau_embedding_dim must be > 0.")
    if args.ddp and args.cpu:
        parser.error("--ddp cannot be combined with --cpu.")
    return args


def main() -> None:
    args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    reports_dir = (repo_root / args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        gpu_count = 0 if args.cpu else get_gpu_count()
        if args.ddp:
            nproc_per_node = args.nproc_per_node if args.nproc_per_node > 0 else gpu_count
            if nproc_per_node <= 1:
                raise RuntimeError("--ddp requires at least 2 CUDA devices.")
            if gpu_count > 0 and nproc_per_node > gpu_count:
                raise RuntimeError(
                    f"--nproc_per_node={nproc_per_node} exceeds available GPUs ({gpu_count})."
                )
            args.nproc_per_node = nproc_per_node
            if args.effective_batch_size > 0:
                if args.effective_batch_size % nproc_per_node != 0:
                    raise RuntimeError(
                        "--effective_batch_size must be divisible by --nproc_per_node."
                    )
                args.batch_size = args.effective_batch_size // nproc_per_node

        default_parallel = 1 if args.ddp else (min(4, gpu_count) if gpu_count > 0 else 1)
        max_parallel = args.max_parallel if args.max_parallel > 0 else default_parallel
        max_parallel = max(1, max_parallel)
        if args.ddp and max_parallel != 1:
            print("DDP uses all selected GPUs per scenario; forcing max_parallel=1.")
            max_parallel = 1

        print(
            f"Running {len(SCENARIOS)} scenarios with max_parallel={max_parallel} "
            f"and gpu_count={gpu_count}."
        )
        if args.ddp:
            print(
                f"DDP enabled with nproc_per_node={args.nproc_per_node}, "
                f"batch_size_per_process={args.batch_size}, "
                f"effective_batch_size={args.batch_size * args.nproc_per_node}."
            )

        gpu_queue: queue.Queue[int] | None = None
        if gpu_count > 0 and not args.ddp:
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
