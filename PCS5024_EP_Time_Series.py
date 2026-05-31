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
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
#import uniplot
import random
from dataclasses import dataclass
from typing import Callable

# --- Configuration ---
DEFAULT_PAST_LEN = 10 * 800
DEFAULT_FUTURE_LEN = 10 * 200
DEFAULT_SLIDING_WINDOW_STEP = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DATA_FILENAME = "data/santos_ssh.csv"
DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01 00:00:00"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200
DATA_REMOVAL_RATIO = 0.3

DEFAULT_POS_ENCODING_DIM = 16
DEFAULT_TIME_SCALE = 60.0

SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

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

    print("Creating sequences and dataloaders...")
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

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_dataset = TensorDataset(
        x_train,
        x_train_timestamps,
        x_train_lengths,
        y_train,
        y_train_timestamps,
        y_train_lengths,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        x_test,
        x_test_timestamps,
        x_test_lengths,
        y_test,
        y_test_timestamps,
        y_test_lengths,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        test_dataloader,
    )


def data_preparation(args: argparse.Namespace) -> PreparedData:
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

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

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

    sample_train = sorted(
        random.sample(
            range(train_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * train_data_scaled.height),
        )
    )
    train_data_scaled = train_data_scaled[sample_train]

    sample_test = sorted(
        random.sample(
            range(test_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * test_data_scaled.height),
        )
    )
    test_data_scaled = test_data_scaled[sample_test]

    train_dataloader, test_dataloader = prepare_dataloaders(
        train_df_features=train_data_scaled,
        test_df_features=test_data_scaled,
        past_len=int(args.past_len),
        future_len=int(args.future_len),
        batch_size=int(args.batch_size),
        sliding_window_step=int(args.sliding_window_step),
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

    def __init__(self, input_size: int, hidden_size: int, pos_encoding_dim: int = DEFAULT_POS_ENCODING_DIM,time_scale: float = DEFAULT_TIME_SCALE,):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pos_encoding_dim = pos_encoding_dim
        self.time_scale = time_scale

        # Encoder recebe: SSH + positional encoding do tempo passado
        self.encoder = nn.GRU(
            input_size + pos_encoding_dim,
            hidden_size,
            batch_first=True,
        )

        # Decoder recebe: positional encoding do tempo futuro
        self.decoder = nn.GRU(
            pos_encoding_dim,
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

        x_timestamps = self._timestamps_to_2d(x_timestamps)

        # Tempos passados relativos à origem da previsão.
        # Como são tempos passados, ficam negativos.
        x_relative_positions = (x_timestamps - origin_timestamp) / self.time_scale

        x_pe = sinusoidal_positional_encoding(
        positions=x_relative_positions,
        dim=self.pos_encoding_dim,
        )

        # Entrada final do encoder:
        # [SSH, PE(t)]
        x_augmented = torch.cat([x, x_pe], dim=-1)

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_augmented, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(x_packed)  # h_n shape: (1, batch_size, hidden_size)
        return h_n

    def decode(self, h_n: torch.Tensor, y_timestamps: torch.Tensor, y_lengths: torch.Tensor, origin_timestamp: torch.Tensor) -> torch.Tensor:
        """Decodes the sequence autoregressively."""
        
        max_target_length = int(y_lengths.max().item())


        y_timestamps = self._timestamps_to_2d(
        y_timestamps[:, :max_target_length]
        )

        # Tempos futuros relativos à origem da previsão.
        # O +1 faz o primeiro ponto futuro ser +1 em vez de 0.
        y_relative_positions = (
            (y_timestamps - origin_timestamp) / self.time_scale
        ) + 1.0

        y_pe = sinusoidal_positional_encoding(
            positions=y_relative_positions,
            dim=self.pos_encoding_dim,
        )

        y_packed = nn.utils.rnn.pack_padded_sequence(
            y_pe,
            y_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )


        out, _ = self.decoder(y_packed, h_n)

        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]

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

    progress_bar = tqdm(dataloader, desc="Training")

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
        inputs = input_features.to(device)
        input_timestamps = input_timestamps.to(device)
        target_timestamps = target_timestamps.to(device)
        targets = target_features.to(device)

        optimizer.zero_grad()

        max_target_length = int(target_lengths.max().item())
        targets = targets[:, :max_target_length]

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

    progress_bar = tqdm(dataloader, desc="Testing")

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
            inputs = input_features.to(device)
            input_timestamps = input_timestamps.to(device)
            target_timestamps = target_timestamps.to(device)
            targets = target_features.to(device)

            max_target_length = int(target_lengths.max().item())
            targets = targets[:, :max_target_length]
            
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


def plot_results(train_losses: list[float], test_losses: list[float], epoch: int) -> None:
    """Plots training and testing loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epoch + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss (MSE)")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()


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
    train_losses: list[float], test_losses: list[float], num_epochs: int
) -> None:
    """Plots the aggregated training and evaluation losses."""

    plot_results(train_losses, test_losses, num_epochs)


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

    print("\n--- Starting Training ---")
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_losses.append(train_loss)
        print(f"Average Training Loss: {train_loss:.4f}")

        eval_result = run_eval_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            norm_statistics=norm_statistics,
        )

        test_loss = eval_result.avg_loss

        test_losses.append(test_loss)
        print(f"Average Test Loss: {test_loss:.4f}")

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

    print("\n--- Training Complete ---")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    prepared_data = data_preparation(args)
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
    ).to(device)
    
    criterion = criterion_without_padding(nn.MSELoss())
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
    model_save_path = pathlib.Path("model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Results ---
    plot_training_history(train_losses, test_losses, args.num_epochs)

    print("Script finished.")


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
        help="Step size for sliding window.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
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
