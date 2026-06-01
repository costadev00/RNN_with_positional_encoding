Você está dentro de um repositório local do EP2 da disciplina PCS5024, Aprendizado Estatístico.

Tarefa:
Audite este repositório e verifique se ele já satisfaz todas as exigências do EP2. Não implemente nada ainda. Primeiro faça uma análise técnica completa, aponte o que já está implementado, o que está parcialmente implementado, o que está faltando e o que provavelmente precisa ser corrigido para atingir nota máxima.

Contexto do EP2:
O trabalho exige implementar e comparar modelos para previsão de séries temporais com dados faltantes, combinando codificação temporal no estilo de Vaswani et al. 2017 com o modelo IQN descrito em Gouttes et al. 2021.

Requisitos mínimos do EP2:
1. Implementar codificação temporal inspirada em Vaswani et al. 2017.
2. As features temporais codificadas devem ser concatenadas à feature SSH, resultando em T + 1 features de entrada.
3. Implementar IQN, Implicit Quantile Networks, conforme Gouttes et al. 2021.
4. Integrar IQN ao pipeline de previsão para produzir estimativas probabilísticas da série temporal.
5. Comparar modelo base e modelo com codificação temporal.
6. Fazer essa comparação tanto com dados completos quanto com diferentes níveis de dados faltantes.
7. Implementar teste de cobertura dos resultados da IQN para avaliar a qualidade dos quantis estimados.
8. Gerar bons plots para o relatório.
9. Entregar código fonte atualizado.
10. Entregar relatório em PDF descrevendo implementação, desafios, resultados e plots.

O que você deve fazer nesta auditoria:
1. Leia a estrutura do projeto.
2. Identifique os arquivos principais.
3. Identifique como os dados são carregados.
4. Identifique se existe tratamento de dados faltantes.
5. Identifique se existe geração artificial de missing data com diferentes taxas.
6. Identifique se existe modelo base de previsão temporal.
7. Identifique se o modelo base usa GRU, LSTM, RNN, Transformer ou outro modelo.
8. Verifique se existe codificação temporal sinusoidal no estilo Vaswani.
9. Verifique se a codificação temporal usa timestamps relativas por janela, e não timestamps absolutas.
10. Verifique se a codificação temporal é concatenada ao SSH no encoder.
11. Verifique se a entrada do modelo realmente vira T + 1 features.
12. Verifique se o decoder também recebe informação temporal futura quando aplicável.
13. Verifique se existe implementação de IQN.
14. Verifique se a IQN usa amostragem de tau em [0, 1].
15. Verifique se a IQN usa embedding de tau, preferencialmente com cos(pi * i * tau), como no modelo de Implicit Quantile Networks.
16. Verifique se a loss de quantis foi implementada corretamente.
17. Verifique se as máscaras de padding são respeitadas na loss e nas métricas.
18. Verifique se o modelo gera quantis como q05, q10, q50, q90 e q95.
19. Verifique se existe avaliação de cobertura, por exemplo:
    coverage_80 = mean(q10 <= y_true <= q90)
    coverage_90 = mean(q05 <= y_true <= q95)
20. Verifique se existe comparação quantitativa entre os modelos.
21. Verifique se existem métricas como MAE, MSE, RMSE e quantile loss.
22. Verifique se os experimentos rodam com dados completos e com dados faltantes, por exemplo missing ratio 0.0, 0.1, 0.3 e 0.5.
23. Verifique se os resultados são salvos de forma reprodutível.
24. Verifique se existem plots suficientes para o relatório.
25. Verifique se há seeds fixadas.
26. Verifique se o código roda em CPU e GPU.
27. Verifique se há instruções claras de execução no README.
28. Verifique se existe ou não relatório em PDF.
29. Verifique se o relatório, caso exista, descreve implementação, desafios, resultados e plots.
30. Verifique se há risco de vazamento de dados entre treino e teste.

Formato da resposta:
Retorne um relatório de auditoria com as seguintes seções:

1. Resumo executivo
Diga claramente:
- O repositório parece pronto para entrega?
- O repositório provavelmente tira 10?
- Quais são os maiores riscos?

2. Checklist dos requisitos do EP2
Crie uma tabela com as colunas:
- Requisito
- Status: OK, Parcial, Ausente ou Incerto
- Evidência no código
- Arquivo/linha
- Comentário técnico

3. Arquivos principais encontrados
Liste os arquivos mais importantes e explique o papel de cada um.

4. Pipeline atual
Explique:
- Como os dados são carregados
- Como as janelas temporais são criadas
- Como os dados faltantes são simulados ou tratados
- Como treino/teste são separados
- Como o modelo é treinado
- Como o modelo é avaliado

5. Modelos encontrados
Para cada modelo encontrado, explique:
- Nome da classe/função
- Arquitetura
- Entrada esperada
- Saída produzida
- Se é determinístico ou probabilístico
- Se usa codificação temporal
- Se usa IQN

6. Auditoria da codificação temporal
Verifique especificamente:
- Se existe codificação sinusoidal
- Se segue a ideia de Vaswani et al.
- Se concatena com SSH
- Se usa T + 1 features
- Se usa tempo relativo por janela
- Se evita timestamp absoluta
- Se o decoder recebe informação temporal do horizonte futuro

7. Auditoria da IQN
Verifique especificamente:
- Se existe tau
- Se tau é amostrado corretamente
- Se existe embedding de tau
- Se existe quantile loss
- Se há múltiplos quantis na inferência
- Se há cobertura empírica
- Se há bandas de incerteza nos plots

8. Auditoria dos experimentos
Verifique:
- Quais missing ratios são testados
- Quais modelos são comparados
- Quais métricas são calculadas
- Se há baseline justo
- Se há comparação com e sem codificação temporal
- Se há comparação com dados completos e incompletos

9. Auditoria dos plots
Liste os plots existentes e diga se são suficientes.
Plots desejáveis:
- Série original com dados faltantes destacados
- Loss de treino/teste
- Previsão do baseline
- Previsão com codificação temporal
- Comparação entre modelos
- Bandas probabilísticas da IQN
- Cobertura nominal versus cobertura empírica
- Métricas por missing ratio

10. Problemas encontrados
Liste os problemas em ordem de prioridade:
- Crítico: impede atender o enunciado
- Alto: pode perder muitos pontos
- Médio: afeta qualidade ou clareza
- Baixo: melhoria de organização

11. Plano de correção
Se houver lacunas, proponha um plano de implementação em etapas:
- Etapa 1: correções mínimas para atender o enunciado
- Etapa 2: melhorias para nota alta
- Etapa 3: melhorias para relatório e visualizações

12. Comandos de execução
Identifique os comandos atuais para rodar o projeto.
Se não existirem, sugira comandos ideais, por exemplo:
python main.py --model_type deterministic --data_removal_ratio 0.3
python main.py --model_type deterministic --use_time_encoding --data_removal_ratio 0.3
python main.py --model_type iqn --use_time_encoding --data_removal_ratio 0.3

13. Veredito final
Classifique o estado do repositório:
- Pronto para entrega
- Quase pronto
- Parcialmente pronto
- Ainda insuficiente

Inclua uma justificativa objetiva.

Importante:
- Não modifique arquivos nesta primeira etapa.
- Não assuma que algo está implementado só pelo nome do arquivo.
- Procure evidências reais no código.
- Cite arquivo e linha sempre que possível.
- Se alguma parte estiver confusa, marque como Incerto.
- Seja rigoroso, porque o objetivo é saber se esse repositório realmente satisfaz o EP2.