# AGENTS.md

## Objetivo do projeto

Este projeto é o EP2 da disciplina PCS5024, Aprendizado Estatístico.

O objetivo é implementar e comparar modelos para previsão de séries temporais com dados faltantes, usando:

1. Modelo base para forecasting de SSH.
2. Codificação temporal inspirada em Vaswani et al. 2017.
3. IQN, Implicit Quantile Networks, conforme Gouttes et al. 2021.
4. Avaliação com dados completos e com diferentes níveis de dados faltantes.
5. Teste de cobertura dos quantis estimados pela IQN.

O alvo principal do forecasting é a série temporal de SSH.

## Requisitos obrigatórios do EP2

Antes de considerar o projeto pronto, verifique se existem evidências reais no código para:

1. Forecasting de SSH.
2. Simulação ou tratamento de dados faltantes.
3. Experimentos com dados completos, por exemplo missing ratio 0.0.
4. Experimentos com dados faltantes, pelo menos missing ratio 0.3.
5. Comparação entre modelo base e modelo com codificação temporal.
6. Codificação temporal sinusoidal inspirada em Vaswani et al. 2017.
7. Entrada do encoder com codificação temporal concatenada ao SSH, resultando em T + 1 features.
8. Uso de timestamps relativos por janela, evitando timestamp absoluta.
9. Decoder recebendo informação temporal futura quando a codificação temporal estiver ativa.
10. Implementação de IQN.
11. Amostragem de tau em [0, 1].
12. Embedding de tau para a IQN.
13. Quantile loss, também chamada de pinball loss.
14. Inferência com múltiplos quantis, por exemplo q05, q10, q50, q90 e q95.
15. Teste de cobertura dos intervalos q10 a q90 e q05 a q95.
16. Métricas agregadas, como MSE, RMSE, MAE e quantile loss.
17. Plots bons para relatório.
18. Relatório ou estrutura de relatório em PDF.

## Como o Codex deve trabalhar

Antes de modificar qualquer arquivo:

1. Leia o README, o enunciado do trabalho e os arquivos principais de treino, avaliação e preparação de dados.
2. Identifique os arquivos principais do projeto.
3. Liste quais requisitos do EP2 já estão atendidos.
4. Liste quais requisitos estão parcialmente atendidos.
5. Liste quais requisitos ainda estão ausentes.
6. Apresente um plano curto antes de implementar.
7. Não reescreva o projeto inteiro se uma modificação incremental resolver.
8. Preserve o funcionamento do baseline atual.

## Arquivos importantes

Priorize:

1. README.md
2. requirements.txt ou pyproject.toml
3. Arquivos de treino
4. Arquivos de avaliação
5. Arquivos de preparação de dados
6. Implementações de modelos
7. Scripts de geração de plots
8. Arquivos de relatório

Evite explorar ou modificar:

1. Caches
2. Checkpoints grandes
3. Outputs antigos
4. Arquivos temporários
5. Notebooks que não sejam essenciais

## Padrões de implementação

Ao modificar código Python:

1. Mantenha compatibilidade com CPU e GPU.
2. Garanta que todos os tensores estejam no mesmo device.
3. Use seeds para reprodutibilidade.
4. Não use o conjunto de teste para escolher hiperparâmetros.
5. Respeite máscaras de padding nas losses e métricas.
6. Salve resultados em arquivos organizados.
7. Salve plots em uma pasta de outputs.
8. Prefira funções pequenas e nomes explícitos.
9. Preserve o baseline determinístico.
10. Separe métricas determinísticas de métricas probabilísticas da IQN.

## Métricas esperadas

Para modelos determinísticos, calcule:

1. MSE
2. RMSE
3. MAE

Para IQN, calcule:

1. Quantile loss
2. MAE da mediana q50
3. RMSE da mediana q50
4. Cobertura 80 por cento, usando q10 a q90
5. Cobertura 90 por cento, usando q05 a q95
6. Largura média dos intervalos

## Plots esperados

Sempre que possível, gere:

1. Série original com dados faltantes destacados.
2. Curva de loss de treino e teste.
3. Forecast do baseline.
4. Forecast com codificação temporal.
5. Comparação baseline versus codificação temporal.
6. Forecast IQN com q50 e bandas q10 a q90 e q05 a q95.
7. Gráfico de cobertura nominal versus cobertura empírica.
8. Métricas por nível de dados faltantes.

## Comandos de validação

Quando modificar código Python, rode pelo menos:

```bash
python -m pytest

SeSe não houver testes, rode um teste rápido de execução, por exemplo:
python main.py --help