# Previsão de SSH com GRU, codificação temporal e IQN

Projeto desenvolvido para o trabalho de PCS5024, com foco em previsão de séries
temporais de altura da superfície do mar (SSH, `sea surface height`) sob dados
faltantes. A implementação compara um GRU base, um GRU com codificação temporal
sinusoidal inspirada em Vaswani et al. (2017), e uma extensão probabilística com
IQN conforme Gouttes et al. (2021).

## Por que este projeto importa

Séries oceanográficas frequentemente têm falhas de observação, intervalos
irregulares e incerteza operacional. Um modelo recorrente que enxerga apenas a
ordem das amostras pode confundir uma sequência regular com uma sequência que
teve pontos removidos. A codificação temporal adiciona a posição relativa das
observações ao modelo, ajudando a GRU a interpretar janelas irregulares. A IQN
complementa essa previsão pontual com quantis e bandas de incerteza, permitindo
avaliar não apenas "qual valor o modelo prevê", mas também "com que cobertura o
modelo expressa sua incerteza".

## O que foi implementado

- `PCS5024_EP_Time_Series.py`: script principal de treino de um cenário.
  - `gru_base`: GRU encoder-decoder determinístico sem codificação temporal.
  - `gru_temporal`: GRU com codificação temporal sinusoidal concatenada à SSH.
  - `gru_temporal_iqn`: GRU temporal com cabeça IQN para previsão probabilística.
- `run_reports.py`: execução oficial do experimento final, com 12 cenários A-L,
  geração de CSVs de resumo, tabela LaTeX e figuras.
- `relatorio.tex`: relatório final em LaTeX com metodologia, desafios,
  resultados e plots.
- `reports_iqn_temporal/`: artefatos finais leves usados no relatório.

A IQN usa `tau ~ U(0, 1)` no treino, embedding cossenoidal de `tau`, combinação
com o estado oculto do decoder e otimização por pinball loss mascarada para
ignorar padding. Na avaliação, os quantis exportados são `q05`, `q10`, `q50`,
`q90` e `q95`; a mediana `q50` também é gravada como `Y_hat` para manter a
compatibilidade com as métricas pontuais.

## Cenários finais

| Cenário | Modelo | Missing | Objetivo |
|---|---|---:|---|
| A | GRU base | 0.0 | baseline ideal |
| B | GRU base | 0.1 | robustez |
| C | GRU base | 0.3 | degradação principal |
| D | GRU base | 0.5 | degradação severa |
| E | GRU temporal | 0.0 | controle |
| F | GRU temporal | 0.1 | robustez |
| G | GRU temporal | 0.3 | recuperação principal |
| H | GRU temporal | 0.5 | robustez severa |
| I | GRU temporal IQN | 0.0 | IQN controle |
| J | GRU temporal IQN | 0.1 | IQN robustez |
| K | GRU temporal IQN | 0.3 | IQN recuperação principal |
| L | GRU temporal IQN | 0.5 | IQN robustez severa |

## Resultados principais

Os resultados completos estão em `reports_iqn_temporal/resumo_resultados.csv` e
no relatório. Em resumo:

- O GRU temporal reduziu o RMSE em todos os níveis de dados faltantes quando
  comparado ao GRU base.
- Os ganhos relativos de RMSE do GRU temporal sobre o GRU base ficaram em torno
  de 22% (`missing=0.0`), 31% (`0.1`), 41% (`0.3`) e 31% (`0.5`).
- A IQN produziu intervalos probabilísticos com cobertura empírica entre 71% e
  79% para o intervalo nominal de 80%, e entre 84% e 89% para o intervalo
  nominal de 90%.
- O custo computacional da IQN é maior, pois o treino usa 8 amostras de `tau`
  por ponto futuro e a avaliação usa 100 amostras por ponto futuro.

## Ambiente utilizado

Execução final realizada em:

- CPU: AMD Ryzen Threadripper PRO 7975WX 32-Cores, 64 threads.
- RAM: 246 GiB.
- GPU: 4x NVIDIA RTX 4000 Ada Generation, aproximadamente 20 GiB por GPU.
- Python: 3.14.4.
- PyTorch: 2.12.0 com CUDA 13.0.
- Treino final: DistributedDataParallel com 4 GPUs, batch por processo 256 e
  batch efetivo 1024.

## Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para rodar em GPU/DDP, instale uma build do PyTorch compatível com a versão de
CUDA e driver da máquina. Em CPU, os mesmos scripts funcionam, mas o experimento
completo pode demorar bastante.

## Como rodar

Validar as CLIs:

```bash
python PCS5024_EP_Time_Series.py --help
python run_reports.py --help
```

Rodar um cenário individual determinístico:

```bash
python PCS5024_EP_Time_Series.py \
  --model_variant gru_temporal \
  --data_removal_ratio 0.3 \
  --output_dir /tmp/pcs5024_gru_temporal \
  --run_name smoke_gru_temporal \
  --num_epochs 1 \
  --hidden_size 16 \
  --batch_size 32
```

Rodar um cenário individual IQN curto:

```bash
python PCS5024_EP_Time_Series.py \
  --model_variant gru_temporal_iqn \
  --data_removal_ratio 0.1 \
  --output_dir /tmp/pcs5024_iqn \
  --run_name smoke_iqn \
  --num_epochs 1 \
  --hidden_size 16 \
  --batch_size 32 \
  --iqn_train_samples 2 \
  --iqn_eval_samples 8
```

Rodar o experimento final completo A-L em uma máquina com 4 GPUs:

```bash
python run_reports.py \
  --reports_dir reports_iqn_temporal \
  --ddp \
  --effective_batch_size 1024 \
  --early_stopping_patience 50 \
  --early_stopping_min_delta 0.0
```

Regenerar apenas tabelas e plots a partir dos artefatos já existentes:

```bash
python run_reports.py --reports_dir reports_iqn_temporal --skip_training
```

Em CPU, use:

```bash
python run_reports.py --reports_dir reports_iqn_temporal --cpu
```

## Relatório

O relatório fonte está em `relatorio.tex`. O PDF final não é versionado no Git;
ele deve ser compilado localmente ou no Overleaf. Para compilar no Overleaf,
envie também a pasta `reports_iqn_temporal/`, pois ela contém as figuras usadas
no texto.

## Checklist do enunciado

- Codificação temporal inspirada em Vaswani et al. (2017): implementada no
  `gru_temporal`, com features temporais concatenadas à SSH.
- IQN conforme Gouttes et al. (2021): implementada em `gru_temporal_iqn`, com
  previsões probabilísticas por quantis.
- Comparação base versus temporal com dados completos e faltantes: cenários A-H.
- Teste de cobertura da IQN: cenários I-L com cobertura 80% e 90%, além de
  largura média dos intervalos.

## Estrutura final

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

Arquivos grandes e regeneráveis, como pesos `.pth`, predições brutas, logs de
treino e reports intermediários, foram removidos/ignorados para manter o
repositório limpo para o push final.
