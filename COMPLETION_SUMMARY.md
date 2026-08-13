# Status da entrega — Tech Challenge Fase 2

Última atualização: 2026-08-12.

Este documento descreve o que existe no repositório e o que foi verificado executando, não o que
se pretendia entregar. Onde há limitação conhecida, ela está escrita.

## Rubrica

| Critério | Peso | Situação | Como verificar |
|---|---|---|---|
| Clean Code e estrutura | 20% | Pacote em `src/`, type hints, ruff limpo com regras E/F/W/I/B/UP/SIM | `ruff check src/ tests/ scripts/` |
| Reprodutibilidade | 20% | `poetry.lock`, `dvc.lock` e `.dvc/` versionados; seed em `params.yaml`; 37 testes, 97% de cobertura | `dvc repro && pytest --cov=src` |
| Docker | 15% | Build multi-stage funcional, usuário não-root, healthcheck; compose com MLflow + API | `docker build -t tc2 . && docker run -p 8000:8000 tc2` |
| DVC + Pipeline | 15% | 3 estágios, `params.yaml` declarado como dependência, métricas reais em `metrics.json` | `dvc repro && dvc metrics show` |
| Modelagem clássica | 10% | LogisticRegression e RandomForest em Pipeline sklearn, `class_weight="balanced"`, seleção por PR-AUC | tabela de métricas no README |
| MLflow + Registry | 20% | 2 runs por execução com 6 métricas, modelo com signature, promoção por alias `@champion` | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

## Limitações conhecidas

1. **O dataset versionado é sintético.** `scripts/generate_sample_data.py` gera 5.000 sessões com
   o mesmo schema do dataset real e um alvo que depende das features, o que permite testar o
   pipeline sem rede. As métricas do README vêm desse dado e **não representam a performance no
   dataset real** — para isso, rode `scripts/download_dataset.py` e `dvc repro`.
2. **Não há DVC remote configurado.** O `.dvc` do dataset está versionado, mas o arquivo em si é
   reconstruído localmente por um dos dois scripts. Compartilhamento real exige
   `dvc remote add -d storage <url>` + `dvc push`.
3. **Sem tuning de hiperparâmetros.** Os valores em `params.yaml` são razoáveis, não otimizados.
   Não há busca em grade nem validação cruzada — a seleção usa um único split 80/20.
4. **Registry local.** O backend padrão é SQLite em arquivo. Para uso multi-usuário, aponte
   `MLFLOW_TRACKING_URI` para um tracking server (o `docker-compose.yml` sobe um).

## Verificado nesta versão

```
ruff check src/ tests/ scripts/   All checks passed
pytest                            37 passed
pytest --cov=src                  97%
dvc repro                         3 estágios, modelo promovido (metrics.json: "promoted": true)
dvc status                        Data and pipelines are up to date
GET  /health                      200 {"status":"healthy","model_loaded":true,...}
POST /predict                     200 {"prediction":1,"probability":0.87}
```

O build Docker é exercitado pela CI (`.github/workflows/ci.yml`), que sobe o container e checa o
`/health`.

## Próximos passos

- [ ] Rodar o pipeline com o dataset real e publicar as métricas resultantes
- [ ] Configurar um DVC remote acessível ao time
- [ ] Tuning de hiperparâmetros (Optuna) com validação cruzada
- [ ] Monitoramento de drift do dado de entrada
