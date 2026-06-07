import argparse
import pathlib
import subprocess
import sys


def detect_cuda_device_count(python_exe: str) -> int:
    command = [
        python_exe,
        "-c",
        (
            "import torch; "
            "print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
        ),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0


def resolve_output_dir(repo_root: pathlib.Path, path_value: str) -> pathlib.Path:
    output_dir = pathlib.Path(path_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the base GRU temporal experiment with 30% data removal and "
            "early stopping. Extra arguments are forwarded to PCS5024_EP_Time_Series.py."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output_dir", default="reports/base_30_gru_temporal")
    parser.add_argument(
        "--nproc_per_node",
        "--nproc-per-node",
        type=int,
        default=0,
        help="Number of GPU processes for distributed training. Use 0 to auto-detect.",
    )
    parser.add_argument(
        "--no_distributed",
        action="store_true",
        help="Disable torch distributed launch and run a single process.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    repo_root = pathlib.Path(__file__).resolve().parent
    output_dir = resolve_output_dir(repo_root, args.output_dir)
    run_name = "base_30_gru_temporal"

    if args.nproc_per_node < 0:
        raise ValueError("--nproc_per_node must be >= 0.")

    detected_gpu_count = detect_cuda_device_count(args.python)
    nproc_per_node = args.nproc_per_node if args.nproc_per_node > 0 else detected_gpu_count
    use_distributed = not args.no_distributed and nproc_per_node > 1

    training_command = [
        str(repo_root / "PCS5024_EP_Time_Series.py"),
        "--model_variant",
        "gru_temporal",
        "--data_removal_ratio",
        "0.3",
        "--run_name",
        run_name,
        "--objective",
        "base 30 missing early stopping",
        "--output_dir",
        str(output_dir),
        "--predictions_csv",
        f"{run_name}_predictions.csv",
        "--history_csv",
        f"{run_name}_history.csv",
        "--metrics_json",
        f"{run_name}_metrics.json",
        "--loss_curve_png",
        f"{run_name}_loss.png",
        "--model_weights_path",
        f"{run_name}_model_weights.pth",
        "--num_epochs",
        "1000",
        "--early_stopping_patience",
        "50",
        "--early_stopping_min_delta",
        "0.0",
        *extra_args,
    ]

    if use_distributed:
        command = [
            args.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(nproc_per_node),
            *training_command,
        ]
        print(
            f"Detected {detected_gpu_count} CUDA GPUs; launching DDP with "
            f"{nproc_per_node} processes.",
            flush=True,
        )
    else:
        command = [args.python, *training_command]
        print(
            f"Detected {detected_gpu_count} CUDA GPUs; launching a single process.",
            flush=True,
        )

    print("Running command:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
