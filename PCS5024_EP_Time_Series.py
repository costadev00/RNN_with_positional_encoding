# PCS5024 - Aprendizado Estatístico - 2026
# EP - Séries Temporais
# Autor: Marcel Rodrigues de Barros (marcel.barros@usp.br)

# Objetivo:
# Implementar e comparar modelos para previsão de séries temporais com dados faltantes,
# combinando codificação temporal no estilo de Vaswani et al. (2017) com o modelo IQN
# descrito em Gouttes et al. (2021), para avaliar o impacto dessas abordagens no desempenho.

# O que os alunos devem implementar:
# 1. Codificação temporal inspirada em Vaswani et al. (2017), com as features temporais
#    codificadas concatenadas à feature de SSH, resultando em T+1 features de entrada.
# 2. IQN (Implicit Quantile Networks) conforme Gouttes et al. (2021), integrado ao pipeline
#    de previsão para produzir estimativas probabilísticas da série temporal.
# 3. Comparação entre o modelo base e o modelo com codificação temporal,
#    tanto com dados completos quanto com diferentes níveis de dados faltantes.
# 4. Teste de cobertura nos resultados da IQN para avaliar a qualidade dos quantis estimados.

# Entregáveis:
# 1. Código-fonte atualizado com a implementação solicitada.
# 2. Relatório em PDF descrevendo a implementação, os desafios enfrentados e os resultados obtidos (plots).

# Priorize boas visualizações!
# Dúvidas devem ser enviadas via fórum no e-Disciplinas.

import datetime
import csv
import json
import os
import pathlib
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import uniplot
import random
from dataclasses import dataclass
from typing import Callable

# --- Configuration ---
DEFAULT_PAST_LEN = 10 * 800
DEFAULT_FUTURE_LEN = 10 * 200
DEFAULT_SLIDING_WINDOW_STEP = 25

DEFAULT_BATCH_SIZE = 256

DEFAULT_HIDDEN_SIZE = 128

DEFAULT_NUM_EPOCHS = 1000

DEFAULT_LEARNING_RATE = 2e-4 #diminui de 2e-4 para 1e-4 para evitar divergência com a adição da codificação temporal
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_DATA_FILENAME = "data/santos_ssh.csv"

DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01 00:00:00"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200
DEFAULT_NUM_WORKERS = 0

DEFAULT_PREDICTIONS_CSV = "test_predictions.csv"
DATA_REMOVAL_RATIO = 0

DEFAULT_POS_ENCODING_DIM = 128
DEFAULT_TIME_SCALE = 1.0

SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def set_global_seed(seed: int) -> None:
    """Sets all random seeds used by this script."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_path(output_dir: str, path_value: str) -> pathlib.Path:
    """Resolves relative output paths against the configured output directory."""

    output_path = pathlib.Path(path_value)
    if not output_path.is_absolute():
        output_path = pathlib.Path(output_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initializes torch distributed when launched with torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return distributed, rank, local_rank, world_size


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def log(message: str) -> None:
    if is_main_process():
        print(message)


def distributed_barrier(device: torch.device) -> None:
    if not is_distributed():
        return
    if device.type == "cuda":
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        dist.barrier(device_ids=[device_id])
    else:
        dist.barrier()


@dataclass(slots=True)
class PreparedData:
    feature_names: list[str]
    norm_statistics: tuple[torch.Tensor, torch.Tensor]
    train_dataloader: DataLoader
    test_dataloader: DataLoader


@dataclass(slots=True)
class EvaluationResult:
    avg_loss: float
    contexts: list[np.ndarray]
    context_timestamps: list[np.ndarray]
    predictions: list[np.ndarray]
    targets: list[np.ndarray]
    target_timestamps: list[np.ndarray]


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    """Loads data from CSV, and sets datetime and feature types.

    Args:
        file_path (pathlib.Path): Path to the CSV file.
    Returns:
        pl.DataFrame: Loaded and preprocessed DataFrame.
    """

    df = pl.read_csv(file_path)
    df = df.with_columns(
        [
            pl.col("datetime").str.to_datetime(
                time_unit="ms",
                strict=True,
                exact=True,
                format="%Y-%m-%d %H:%M:%S+00:00",
            ),
        ]
        + [pl.col(f).cast(pl.Float32) for f in df.columns if f != "datetime"]
    )

    return df


def split_data(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits the data into training and testing sets based on the split date.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
        split_date (datetime.datetime): Date to split the data.
    Returns:
        tuple: Training and testing DataFrames.
    """

    train_df = df.filter(pl.col("datetime") < split_date)
    test_df = df.filter(pl.col("datetime") >= split_date)

    log(f"Train set size: {len(train_df)}")
    log(f"Test set size: {len(test_df)}")

    return train_df, test_df


def create_sequences(
    df: pl.DataFrame,
    past_len: int,
    future_len: int,
    step: int = 1,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Creates windows using a sliding window approach.

    Args:
        data (pl.DataFrame): DataFrame containing the data.
        past_len (int): Length of the past sequence in minutes.
        future_len (int): Length of the future sequence in minutes.
        step (int): Step size for the sliding window in minutes.

    Returns:
        tuple: Arrays of past and future sequences of features and timestamps
    """

    xs, xs_timestamps, xs_lengths, ys, ys_timestamps, ys_lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    datetime_values = df.get_column("datetime").to_numpy().astype(np.float32)
    start_time = float(np.min(datetime_values)) + float(past_len)
    stop_time = float(np.max(datetime_values)) - float(future_len)
    observer_times = np.arange(start_time, stop_time, step)
    for ot in observer_times:
        lb = df["datetime"].search_sorted(ot - past_len, side="left")
        obs = df["datetime"].search_sorted(ot, side="left")
        ub = df["datetime"].search_sorted(ot + future_len, side="left")
        x = df[lb:obs].select(pl.exclude("datetime")).to_numpy()
        x_timestamps = df[lb:obs].select(pl.col("datetime")).to_numpy()
        x_length = x.shape[0]
        y = df[obs:ub].select(pl.exclude("datetime")).to_numpy()
        y_timestamps = df[obs:ub].select(pl.col("datetime")).to_numpy()
        y_length = y.shape[0]

        if x_length == 0 or y_length == 0:
            continue
        xs.append(torch.tensor(x))
        xs_timestamps.append(torch.tensor(x_timestamps))
        xs_lengths.append(torch.tensor(x_length))
        ys.append(torch.tensor(y))
        ys_timestamps.append(torch.tensor(y_timestamps))
        ys_lengths.append(torch.tensor(y_length))
    return (
        torch.nn.utils.rnn.pad_sequence(xs, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(xs_timestamps, batch_first=True),
        torch.stack(xs_lengths),
    ), (
        torch.nn.utils.rnn.pad_sequence(ys, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(ys_timestamps, batch_first=True),
        torch.stack(ys_lengths),
    )


def prepare_dataloaders(
    train_df_features: pl.DataFrame,
    test_df_features: pl.DataFrame,
    past_len: int,
    future_len: int,
    batch_size: int,
    sliding_window_step: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Creates sequences and prepares PyTorch DataLoaders.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame.
        past_len (int): Length of the past sequence.
        future_len (int): Length of the future sequence.
        batch_size (int): Batch size for DataLoader.
        sliding_window_step (int): Step size for sliding window.

    Returns:
        tuple: Training and testing DataLoaders.
    """

    log("Creating sequences and dataloaders...")
    (x_train, x_train_timestamps, x_train_lengths), (
        y_train,
        y_train_timestamps,
        y_train_lengths,
    ) = create_sequences(
        df=train_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )
    (x_test, x_test_timestamps, x_test_lengths), (
        y_test,
        y_test_timestamps,
        y_test_lengths,
    ) = create_sequences(
        df=test_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )

    log(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    log(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_dataset = TensorDataset(
        x_train,
        x_train_timestamps,
        x_train_lengths,
        y_train,
        y_train_timestamps,
        y_train_lengths,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED,
        )
        if distributed
        else None
    )
    pin_memory = torch.cuda.is_available()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dataset = TensorDataset(
        x_test,
        x_test_timestamps,
        x_test_lengths,
        y_test,
        y_test_timestamps,
        y_test_lengths,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_dataloader,
        test_dataloader,
    )


def data_preparation(
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
    world_size: int,
) -> PreparedData:
    """Loads, splits, scales, and converts the data into dataloaders."""

    file_path = pathlib.Path(args.data_filename)
    split_date = datetime.datetime.fromisoformat(args.split_date)
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns)

    train_df, test_df = split_data(df=df, split_date=split_date)

    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()
    norm_statistics = (
        torch.tensor(train_mean.row(0), dtype=torch.float32),
        torch.tensor(train_std.row(0), dtype=torch.float32),
    )

    log(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    train_data_scaled = train_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )
    test_data_scaled = test_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )

    train_data_scaled = train_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    )
    test_data_scaled = test_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    )

    data_removal_ratio = float(args.data_removal_ratio)
    if not 0.0 <= data_removal_ratio < 1.0:
        raise ValueError("--data_removal_ratio must be in the interval [0.0, 1.0).")

    if data_removal_ratio > 0.0:
        rng = random.Random(int(args.seed))
        train_keep_count = max(
            1, int(round((1.0 - data_removal_ratio) * train_data_scaled.height))
        )
        test_keep_count = max(
            1, int(round((1.0 - data_removal_ratio) * test_data_scaled.height))
        )
        sample_train = sorted(rng.sample(range(train_data_scaled.height), train_keep_count))
        sample_test = sorted(rng.sample(range(test_data_scaled.height), test_keep_count))
        train_data_scaled = train_data_scaled[sample_train]
        test_data_scaled = test_data_scaled[sample_test]
        log(
            "Applied missing-data simulation: "
            f"ratio={data_removal_ratio:.2f}, "
            f"train_kept={train_keep_count}/{train_df.height}, "
            f"test_kept={test_keep_count}/{test_df.height}"
        )
    else:
        log("Applied missing-data simulation: ratio=0.00, all rows kept.")

    train_dataloader, test_dataloader = prepare_dataloaders(
        train_df_features=train_data_scaled,
        test_df_features=test_data_scaled,
        past_len=int(args.past_len),
        future_len=int(args.future_len),
        batch_size=int(args.batch_size),
        sliding_window_step=int(args.sliding_window_step),
        num_workers=int(args.num_workers),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    return PreparedData(
        feature_names=feature_names,
        norm_statistics=norm_statistics,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

def sinusoidal_positional_encoding(
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """
    positions: tensor com shape (batch, seq_len) ou (batch, seq_len, 1)
    dim: dimensão do positional encoding
    """

    if positions.dim() == 3 and positions.size(-1) == 1:
        positions = positions.squeeze(-1)

    positions = positions.float()

    half_dim = (dim + 1) // 2

    div_term = torch.exp(
        torch.arange(half_dim, device=positions.device, dtype=torch.float32)
        * (-np.log(max_period) / max(half_dim - 1, 1))
    )

    angles = positions.unsqueeze(-1) * div_term

    pe = torch.empty(
        *positions.shape,
        dim,
        device=positions.device,
        dtype=torch.float32,
    )

    pe[..., 0::2] = torch.sin(angles[..., : pe[..., 0::2].shape[-1]])
    pe[..., 1::2] = torch.cos(angles[..., : pe[..., 1::2].shape[-1]])

    return pe


# --- Model Definition ---
class ARModel(nn.Module):
    """Autoregressive RNN Model using GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        pos_encoding_dim: int = DEFAULT_POS_ENCODING_DIM,
        time_scale: float = DEFAULT_TIME_SCALE,
        use_time_encoding: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pos_encoding_dim = pos_encoding_dim
        self.time_scale = time_scale
        self.use_time_encoding = use_time_encoding

        encoder_input_size = (
            input_size + pos_encoding_dim if use_time_encoding else input_size
        )
        decoder_input_size = pos_encoding_dim if use_time_encoding else input_size

        # Modelo temporal: encoder recebe [SSH, PE(t)].
        # Modelo base: encoder recebe apenas SSH.
        self.encoder = nn.GRU(
            encoder_input_size,
            hidden_size,
            batch_first=True,
        )

        # Modelo temporal: decoder recebe PE(t futuro).
        # Modelo base: decoder recebe zeros, sem informação temporal explícita.
        self.decoder = nn.GRU(
            decoder_input_size,
            hidden_size,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, input_size)
    
    @staticmethod
    def _timestamps_to_2d(timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.dim() == 3 and timestamps.size(-1) == 1:
            return timestamps.squeeze(-1)
        return timestamps

    def encode(self, x: torch.Tensor, x_timestamps: torch.Tensor, x_lengths: torch.Tensor, origin_timestamp: torch.Tensor) -> torch.Tensor:
        """Encodes the input sequence."""

        if self.use_time_encoding:
            x_timestamps = self._timestamps_to_2d(x_timestamps)

            # Tempos passados relativos à origem da previsão.
            # Como são tempos passados, ficam negativos.
            x_relative_positions = (x_timestamps - origin_timestamp) / self.time_scale

            x_pe = sinusoidal_positional_encoding(
                positions=x_relative_positions,
                dim=self.pos_encoding_dim,
            )

            encoder_inputs = torch.cat([x, x_pe], dim=-1)
        else:
            encoder_inputs = x

        x_packed = nn.utils.rnn.pack_padded_sequence(
            encoder_inputs, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(x_packed)  # h_n shape: (1, batch_size, hidden_size)
        return h_n

    def decode(self, h_n: torch.Tensor, y_timestamps: torch.Tensor, y_lengths: torch.Tensor, origin_timestamp: torch.Tensor) -> torch.Tensor:
        """Decodes the sequence autoregressively."""
        
        max_target_length = y_timestamps.size(1)


        if self.use_time_encoding:
            y_timestamps = self._timestamps_to_2d(
                y_timestamps[:, :max_target_length]
            )

            # Tempos futuros relativos à origem da previsão.
            # O +1 faz o primeiro ponto futuro ser +1 em vez de 0.
            y_relative_positions = (
                (y_timestamps - origin_timestamp) / self.time_scale
            ) + 1.0

            decoder_inputs = sinusoidal_positional_encoding(
                positions=y_relative_positions,
                dim=self.pos_encoding_dim,
            )
        else:
            decoder_inputs = torch.zeros(
                h_n.size(1),
                max_target_length,
                self.input_size,
                device=h_n.device,
                dtype=h_n.dtype,
            )

        y_packed = nn.utils.rnn.pack_padded_sequence(
            decoder_inputs,
            y_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )


        out, _ = self.decoder(y_packed, h_n)

        out = nn.utils.rnn.pad_packed_sequence(
            out,
            batch_first=True,
            total_length=max_target_length,
        )[0]

        y_hat = self.linear(out)

        return y_hat

    def forward(
        self, x: torch.Tensor, x_timestamps: torch.Tensor, x_lengths: torch.Tensor, y_timestamps: torch.Tensor, y_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: encode the past, decode the future."""
        
        y_timestamps_2d = self._timestamps_to_2d(y_timestamps)

         # Origem temporal = primeiro instante futuro
        origin_timestamp = y_timestamps_2d[:, :1]
        
        h_n = self.encode(x=x,
        x_timestamps=x_timestamps,
        x_lengths=x_lengths,
        origin_timestamp=origin_timestamp,
        )
        
        output_seq = self.decode(h_n=h_n,
        y_timestamps=y_timestamps,
        y_lengths=y_lengths,
        origin_timestamp=origin_timestamp,
        )

        return output_seq


# --- Training and Evaluation ---


def run_train_epoch(
    model: ARModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    model.train()

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        disable=not is_main_process() or os.environ.get("DISABLE_TQDM") == "1",
    )

    losses = []
    # Use enumerate to get batch index for plotting
    for (
        input_features,
        input_timestamps,
        input_lengths,
        target_features,
        target_timestamps,
        target_lengths,
    ) in progress_bar:
        using_data_parallel = isinstance(model, nn.DataParallel)

        optimizer.zero_grad()

        max_target_length = int(target_lengths.max().item())
        targets = target_features[:, :max_target_length].to(device)
        target_timestamps = target_timestamps[:, :max_target_length]
        if using_data_parallel:
            inputs = input_features
            input_timestamps = input_timestamps
        else:
            inputs = input_features.to(device)
            input_timestamps = input_timestamps.to(device)
            target_timestamps = target_timestamps.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

        outputs = model(
            inputs,
            input_timestamps,
            input_lengths,
            target_timestamps,
            target_lengths,
        )

        loss = criterion(outputs, targets, target_lengths)  # LOSS COMPUTATION

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().item())

    return float(np.mean(losses))


def denormalize_batch(
    batch: np.ndarray,
    norm_statistics: tuple[torch.Tensor, torch.Tensor],
) -> np.ndarray:
    """Converts a normalized batch back to the original data scale."""

    norm_mean, norm_std = norm_statistics
    return batch * norm_std.cpu().numpy() + norm_mean.cpu().numpy()


def run_eval_epoch(
    model: ARModel,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    norm_statistics: tuple[torch.Tensor, torch.Tensor],
) -> EvaluationResult:
    model.eval()

    progress_bar = tqdm(
        dataloader,
        desc="Testing",
        disable=not is_main_process() or os.environ.get("DISABLE_TQDM") == "1",
    )

    total_loss = 0.0
    num_batches = 0
    all_contexts: list[np.ndarray] = []
    all_context_timestamps: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_target_timestamps: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for (
            input_features,
            input_timestamps,
            input_lengths,
            target_features,
            target_timestamps,
        target_lengths,
    ) in progress_bar:
        with torch.no_grad():
            using_data_parallel = isinstance(model, nn.DataParallel)

            max_target_length = int(target_lengths.max().item())
            targets = target_features[:, :max_target_length].to(device)
            target_timestamps = target_timestamps[:, :max_target_length]
            if using_data_parallel:
                inputs = input_features
                input_timestamps = input_timestamps
            else:
                inputs = input_features.to(device)
                input_timestamps = input_timestamps.to(device)
                target_timestamps = target_timestamps.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
            
            predictions = model(
                inputs,
                input_timestamps,
                input_lengths,
                target_timestamps,
                target_lengths,
            )
            
            loss = criterion(predictions, targets, target_lengths)  # LOSS COMPUTATION
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

            contexts = [
                denormalize_batch(
                    inputs[i, : int(input_lengths[i].item())].cpu().numpy(),
                    norm_statistics,
                )
                for i in range(inputs.size(0))
            ]
            input_timestamps = [
                input_timestamps[i, : int(input_lengths[i].item())].cpu().numpy()
                for i in range(input_timestamps.size(0))
            ]
            targets = [
                denormalize_batch(
                    targets[i, : int(target_lengths[i].item())].cpu().numpy(),
                    norm_statistics,
                )
                for i in range(targets.size(0))
            ]
            target_timestamps = [
                target_timestamps[i, : int(target_lengths[i].item())].cpu().numpy()
                for i in range(target_timestamps.size(0))
            ]
            
            predictions = [
                denormalize_batch(
                    predictions[i, : int(target_lengths[i].item())].cpu().numpy(),
                    norm_statistics,
                )
                for i in range(predictions.size(0))
            ]

            all_contexts += contexts
            all_context_timestamps += input_timestamps
            all_targets += targets
            all_target_timestamps += target_timestamps
            all_predictions += predictions

    avg_loss = total_loss / num_batches
    return EvaluationResult(
        avg_loss=avg_loss,
        contexts=all_contexts,
        context_timestamps=all_context_timestamps,
        predictions=all_predictions,
        targets=all_targets,
        target_timestamps=all_target_timestamps,
    )


def export_test_predictions_csv(
    eval_result: EvaluationResult,
    output_path: pathlib.Path,
) -> None:
    """Exports denormalized test targets, predictions, and cumulative mean loss."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["window_id", "step", "Y", "Y_hat", "loss", "loss_media_acumulada"],
        )
        writer.writeheader()

        cumulative_loss = 0.0
        row_count = 0
        for window_id, (target, prediction) in enumerate(
            zip(eval_result.targets, eval_result.predictions)
        ):
            target = np.asarray(target)
            prediction = np.asarray(prediction)
            for step in range(target.shape[0]):
                y = float(target[step, 0])
                y_hat = float(prediction[step, 0])
                loss = (y - y_hat) ** 2
                cumulative_loss += loss
                row_count += 1
                writer.writerow(
                    {
                        "window_id": window_id,
                        "step": step,
                        "Y": y,
                        "Y_hat": y_hat,
                        "loss": loss,
                        "loss_media_acumulada": cumulative_loss / row_count,
                    }
                )

    print(f"Test predictions CSV saved to {output_path}")


def calculate_regression_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    """Calculates deterministic forecasting metrics on the original SSH scale."""

    y_true = np.concatenate([np.asarray(target)[:, 0] for target in eval_result.targets])
    y_pred = np.concatenate(
        [np.asarray(prediction)[:, 0] for prediction in eval_result.predictions]
    )
    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(y_pred - y_true))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "num_points": float(y_true.size),
        "num_windows": float(len(eval_result.targets)),
    }


def export_training_history_csv(
    train_losses: list[float],
    test_losses: list[float],
    output_path: pathlib.Path,
) -> None:
    """Exports train/test loss per epoch."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "test_loss"])
        writer.writeheader()
        for epoch, (train_loss, test_loss) in enumerate(
            zip(train_losses, test_losses), start=1
        ):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )

    log(f"Training history CSV saved to {output_path}")


def export_metrics_json(metrics: dict[str, object], output_path: pathlib.Path) -> None:
    """Exports final metrics and run metadata as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as json_file:
        json.dump(metrics, json_file, indent=2, sort_keys=True)

    log(f"Metrics JSON saved to {output_path}")


def plot_results(
    train_losses: list[float],
    test_losses: list[float],
    epoch: int,
    hyperparameters: dict[str, object],
    output_path: pathlib.Path,
) -> None:
    """Plots training and testing loss curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, epoch + 1), train_losses, label="Training Loss")
    ax.plot(range(1, epoch + 1), test_losses, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Loss (MSE)")
    ax.set_title("Training and Test Loss Over Epochs")
    ax.legend()
    ax.grid(True)

    hyperparameter_text = "\n".join(
        f"{name}: {value}" for name, value in hyperparameters.items()
    )
    ax.text(
        0.98,
        0.98,
        hyperparameter_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()
    log(f"Loss curve saved to {output_path}")


def plot_epoch_results(
    epoch: int,
    eval_result: EvaluationResult,
    view_size: int,
) -> None:
    """Plots the per-epoch prediction example and loss curves."""

    window_id = len(eval_result.contexts) // 2
    example_context = eval_result.contexts[window_id][-view_size:]
    example_context_timestamps = eval_result.context_timestamps[window_id][
        -view_size:
    ]
    example_target = eval_result.targets[window_id]
    example_target_timestamps = eval_result.target_timestamps[window_id]
    example_prediction = eval_result.predictions[window_id]

    example_context = example_context[:, 0]
    example_context_timestamps = example_context_timestamps[:, 0]
    example_target = example_target[:, 0]
    example_target_timestamps = example_target_timestamps[:, 0]
    example_prediction = example_prediction[:, 0]

    #uniplot.plot(
       # ys=[
        #    example_target,
         #   example_context,
          #  example_prediction,
       # ],
        #xs=[
         #   example_target_timestamps,
          #  example_context_timestamps,
           # example_target_timestamps,
        #],
        #color=True,
        #legend_labels=["Target", "Context", "Prediction"],
        #title=f"Epoch: {epoch}, Eval Element: {window_id}, Loss: {eval_result.avg_loss:.4f}",
        #height=15,
        #lines=True,
    #)


def plot_training_history(
    train_losses: list[float],
    test_losses: list[float],
    num_epochs: int,
    hyperparameters: dict[str, object],
    output_path: pathlib.Path,
) -> None:
    """Plots the aggregated training and evaluation losses."""

    plot_results(train_losses, test_losses, num_epochs, hyperparameters, output_path)


def training_loop(
    model: ARModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    norm_statistics: tuple[torch.Tensor, torch.Tensor],
    num_epochs: int,
    view_size: int,
) -> tuple[list[float], list[float]]:
    """Runs the epoch loop and collects training metrics."""

    log("\n--- Starting Training ---")
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        log(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        if is_distributed():
            train_loss_tensor = torch.tensor(train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = float(train_loss_tensor.item() / dist.get_world_size())
        train_losses.append(train_loss)
        log(f"Average Training Loss: {train_loss:.4f}")

        distributed_barrier(device)

        if is_main_process():
            eval_model = model.module if is_distributed() and hasattr(model, "module") else model
            eval_result = run_eval_epoch(
                model=eval_model,
                dataloader=test_dataloader,
                criterion=criterion,
                device=device,
                norm_statistics=norm_statistics,
            )

            test_loss = eval_result.avg_loss

            test_losses.append(test_loss)
            log(f"Average Test Loss: {test_loss:.4f}")

            plot_epoch_results(
                epoch=epoch,
                eval_result=eval_result,
                view_size=view_size,
            )

            #uniplot.plot(
             #   ys=[train_losses, test_losses],
              #  xs=[np.arange(1, epoch + 1)] * 2,
               # color=True,
                #legend_labels=["Train Loss", "Test Loss"],
                #title=f"Epoch: {epoch} Loss Curves",
            #)

        distributed_barrier(device)

    log("\n--- Training Complete ---")
    return train_losses, test_losses


def criterion_without_padding(loss_func:torch.nn.Module) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Computes the loss for a batch, masking out padded elements."""


    def inner_func(
        predictions: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.arange(predictions.size(1), device=predictions.device).unsqueeze(0) < lengths.to(predictions.device).unsqueeze(1)
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        return loss_func(masked_predictions, masked_targets)
    
    return inner_func

def main(args: argparse.Namespace) -> None:
    """Main function to run the training and evaluation."""
    set_global_seed(int(args.seed))
    distributed, rank, local_rank, world_size = setup_distributed()
    cpu_count = os.cpu_count() or 1
    threads_per_process = (
        int(args.torch_threads)
        if int(args.torch_threads) > 0
        else max(1, cpu_count // world_size)
    )
    torch.set_num_threads(threads_per_process)
    torch.set_num_interop_threads(max(1, min(8, threads_per_process)))

    device = torch.device(
        f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    # device = torch.device("cpu")
    log(f"Using device: {device}")
    log(f"CPU threads per process: {torch.get_num_threads()}")
    log(f"CUDA devices available: {cuda_count}")
    if distributed:
        log(f"Using DistributedDataParallel with {world_size} GPUs")
    log(
        "Hyperparameters: "
        f"model_variant={args.model_variant}, "
        f"data_removal_ratio={args.data_removal_ratio}, "
        f"batch_size_per_process={args.batch_size}, "
        f"effective_batch_size={args.batch_size * world_size}, "
        f"hidden_size={args.hidden_size}, "
        f"pos_encoding_dim={args.pos_encoding_dim}, "
        f"learning_rate={args.learning_rate}, "
        f"weight_decay={args.weight_decay}, "
        f"sliding_window_step={args.sliding_window_step}"
    )

    prepared_data = data_preparation(
        args=args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    feature_names = prepared_data.feature_names
    norm_statistics = prepared_data.norm_statistics
    train_dataloader = prepared_data.train_dataloader
    test_dataloader = prepared_data.test_dataloader

    # --- Model Setup ---
    input_size = len(feature_names)
    
    model = ARModel(
        input_size=input_size,
        hidden_size=args.hidden_size,
        pos_encoding_dim=args.pos_encoding_dim,
        time_scale=args.time_scale,
        use_time_encoding=args.model_variant == "gru_temporal",
    ).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    elif cuda_count > 1:
        log("Multiple GPUs detected; launch with torchrun to use all GPUs safely.")
    
    criterion = criterion_without_padding(nn.MSELoss())
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    view_size = int(args.past_view_size)
    train_losses, test_losses = training_loop(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        norm_statistics=norm_statistics,
        num_epochs=args.num_epochs,
        view_size=view_size,
    )

    # --- Save Model ---
    if is_main_process():
        model_save_path = resolve_output_path(args.output_dir, args.model_weights_path)
        predictions_csv_path = resolve_output_path(args.output_dir, args.predictions_csv)
        history_csv_path = resolve_output_path(args.output_dir, args.history_csv)
        metrics_json_path = resolve_output_path(args.output_dir, args.metrics_json)
        loss_curve_path = resolve_output_path(args.output_dir, args.loss_curve_png)

        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), model_save_path)
        log(f"Model saved to {model_save_path}")

        # --- Results ---
        eval_model = model.module if is_distributed() and hasattr(model, "module") else model
        final_eval_result = run_eval_epoch(
            model=eval_model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            norm_statistics=norm_statistics,
        )
        export_test_predictions_csv(
            eval_result=final_eval_result,
            output_path=predictions_csv_path,
        )
        export_training_history_csv(
            train_losses=train_losses,
            test_losses=test_losses,
            output_path=history_csv_path,
        )
        regression_metrics = calculate_regression_metrics(final_eval_result)
        metrics = {
            "run_name": args.run_name,
            "model_variant": args.model_variant,
            "data_removal_ratio": float(args.data_removal_ratio),
            "objective": args.objective,
            "seed": int(args.seed),
            "num_epochs": int(args.num_epochs),
            "batch_size": int(args.batch_size),
            "hidden_size": int(args.hidden_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "past_len": int(args.past_len),
            "future_len": int(args.future_len),
            "sliding_window_step": int(args.sliding_window_step),
            "pos_encoding_dim": int(args.pos_encoding_dim),
            "time_scale": float(args.time_scale),
            "final_train_loss": float(train_losses[-1]),
            "final_test_loss": float(final_eval_result.avg_loss),
            "best_test_loss": float(min(test_losses)),
            **regression_metrics,
        }
        export_metrics_json(metrics=metrics, output_path=metrics_json_path)
        plot_training_history(
            train_losses,
            test_losses,
            args.num_epochs,
            hyperparameters={
                "MODEL": args.model_variant,
                "BATCH_SIZE": args.batch_size,
                "NUM_EPOCHS": args.num_epochs,
                "LEARNING_RATE": args.learning_rate,
                "DATA_REMOVAL_RATIO": args.data_removal_ratio,
                "DEFAULT_POS_ENCODING_DIM": args.pos_encoding_dim,
            },
            output_path=loss_curve_path,
        )

        log("Script finished.")

    if is_distributed():
        distributed_barrier(device)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive RNN Training Script")
    parser.add_argument(
        "--past_len",
        type=int,
        default=DEFAULT_PAST_LEN,
        help="Length of past sequence input.",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=DEFAULT_FUTURE_LEN,
        help="Length of future sequence to predict.",
    )
    parser.add_argument(
        "--sliding_window_step",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_STEP,
        help="Step size for sliding window. Lower values create more training windows.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training. With torchrun/DDP this is per GPU process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader worker processes. Keep 0 because the dataset is already materialized as tensors.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="Number of intra-op/inter-op CPU threads. Use 0 to infer automatically.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in RNN.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay regularization for Adam.",
    )
    parser.add_argument(
        "--data_filename",
        type=str,
        default=DEFAULT_DATA_FILENAME,
        help="Filename to save/load data.",
    )
    parser.add_argument(
        "--split_date",
        type=str,
        default=DEFAULT_TRAIN_TEST_SPLIT_DATE,
        help="Date string for train/test split.",
    )
    parser.add_argument(
        "--past_view_size",
        type=int,
        default=DEFAULT_PAST_PLOT_VIEW_SIZE,
        help="Number of past steps to show in uniplot.",
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        default=DEFAULT_PREDICTIONS_CSV,
        help="CSV file to write test targets, predictions, and per-point loss.",
    )
    parser.add_argument(
        "--history_csv",
        type=str,
        default="training_history.csv",
        help="CSV file to write training and test losses per epoch.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="metrics.json",
        help="JSON file to write final metrics and run metadata.",
    )
    parser.add_argument(
        "--loss_curve_png",
        type=str,
        default="loss_curve.png",
        help="PNG file to write the training and test loss plot.",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="model_weights.pth",
        help="Path to write trained model weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Base directory for relative output paths.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Human-readable name for this experiment run.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="",
        help="Short objective label for this experiment run.",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["gru_base", "gru_temporal"],
        default="gru_temporal",
        help="GRU model variant to train.",
    )
    parser.add_argument(
        "--data_removal_ratio",
        type=float,
        default=DATA_REMOVAL_RATIO,
        help="Fraction of rows to remove from train and test data to simulate missing observations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for initialization, dataloader shuffling, and missing-data simulation.",
    )

    parser.add_argument(
        "--pos_encoding_dim",
        type=int,
        default=DEFAULT_POS_ENCODING_DIM,
        help="Dimension of the sinusoidal temporal encoding.",
    )

    parser.add_argument(
        "--time_scale",
        type=float,
        default=DEFAULT_TIME_SCALE,
        help="Scale used to convert relative timestamps before sinusoidal encoding. Use 60.0 for hours if timestamps are in minutes.",
    )

    args = parser.parse_args()
    main(args)
