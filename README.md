# Previsao de SSH com GRU, codificacao temporal e IQN

Este repositorio contem o codigo do EP2 de PCS5024 para previsao de series
temporais de SSH (`sea surface height`) com tres variantes:

- `gru_base`: GRU encoder-decoder sem codificacao temporal explicita.
- `gru_temporal`: GRU com codificacao temporal sinusoidal.
- `gru_temporal_iqn`: GRU temporal com IQN para previsao probabilistica por
  quantis.

O codigo roda em CPU ou GPU CUDA. A maquina usada para treinar os resultados
finais do relatorio foi a nossa maquina de treino; isso e apenas uma referencia
do experimento, nao um requisito para executar o projeto.

## Requisitos

- Python com uma versao compativel com a sua instalacao do PyTorch.
- `pip`.
- Arquivo de dados em `data/santos_ssh.csv`.
- GPU CUDA e opcional. Sem CUDA, o PyTorch usa CPU automaticamente.

## Instalacao

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para usar GPU, instale uma versao do PyTorch compativel com o CUDA e o driver da
sua maquina. Para CPU, as dependencias de `requirements.txt` sao suficientes.
Nos comandos abaixo, use `python3` em Linux/macOS. No Windows, troque `python3`
por `python`.

## Teste rapido

Verifique se os scripts abrem corretamente:

```bash
python3 PCS5024_EP_Time_Series.py --help
python3 run_reports.py --help
```

Rode um treino curto deterministico:

```bash
python3 PCS5024_EP_Time_Series.py \
  --model_variant gru_temporal \
  --data_removal_ratio 0.3 \
  --output_dir outputs/smoke_gru_temporal \
  --run_name smoke_gru_temporal \
  --num_epochs 1 \
  --hidden_size 16 \
  --batch_size 32 \
  --num_workers 0
```

Rode um treino curto com IQN:

```bash
python3 PCS5024_EP_Time_Series.py \
  --model_variant gru_temporal_iqn \
  --data_removal_ratio 0.1 \
  --output_dir outputs/smoke_iqn \
  --run_name smoke_iqn \
  --num_epochs 1 \
  --hidden_size 16 \
  --batch_size 32 \
  --num_workers 0 \
  --iqn_train_samples 2 \
  --iqn_eval_samples 8
```

Esses comandos servem apenas para validar a instalacao. Eles nao reproduzem os
resultados finais do relatorio.

## Experimento completo

Para rodar os 12 cenarios A-L e gerar tabelas e figuras:

```bash
python3 run_reports.py \
  --reports_dir reports_iqn_temporal \
  --early_stopping_patience 50 \
  --early_stopping_min_delta 0.0
```

Em uma maquina sem GPU, o comando roda em CPU. Para forcar CPU mesmo quando
houver CUDA disponivel:

```bash
python3 run_reports.py \
  --reports_dir reports_iqn_temporal \
  --cpu \
  --early_stopping_patience 50 \
  --early_stopping_min_delta 0.0
```

Em uma maquina com multiplas GPUs CUDA, e possivel usar DDP:

```bash
python3 run_reports.py \
  --reports_dir reports_iqn_temporal \
  --ddp \
  --effective_batch_size 1024 \
  --early_stopping_patience 50 \
  --early_stopping_min_delta 0.0
```

O experimento completo pode demorar bastante em CPU. Para depuracao em qualquer
maquina, reduza `--num_epochs`, `--hidden_size`, `--batch_size`,
`--iqn_train_samples` e `--iqn_eval_samples`.

## Regenerar apenas tabelas e figuras

Se os artefatos em `reports_iqn_temporal/` ja existirem, rode:

```bash
python3 run_reports.py \
  --reports_dir reports_iqn_temporal \
  --skip_training
```

## Saidas geradas

- `outputs/...`: resultados de treinos individuais.
- `reports_iqn_temporal/`: metricas, tabelas, figuras e arquivos usados pelo
  relatorio final.
- `relatorio.tex`: fonte LaTeX do relatorio.

## Estrutura

```text
.
├── PCS5024_EP_Time_Series.py
├── run_reports.py
├── relatorio.tex
├── README.md
├── requirements.txt
├── data/
├── articles/
└── reports_iqn_temporal/
```
