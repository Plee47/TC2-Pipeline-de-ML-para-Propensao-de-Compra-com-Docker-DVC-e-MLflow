# Tech Challenge Fase 2 — Online Shoppers Purchasing Intention

[![CI](https://github.com/Plee47/TC2-Pipeline-de-ML-para-Propensao-de-Compra-com-Docker-DVC-e-MLflow/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-managed-brightgreen.svg)](https://python-poetry.org/)
[![DVC](https://img.shields.io/badge/dvc-versioned-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/mlflow-tracked-red.svg)](https://mlflow.org/)

Pipeline de Machine Learning Engineering para prever a propensão de compra de sessões em e-commerce:
dados versionados com DVC, experimentos rastreados no MLflow, modelo servido por uma API FastAPI,
tudo containerizado.

## Como funciona

```
data/raw/*.csv          (DVC)
        │
        ▼  preprocess   split estratificado 80/20, sem transformar nada
data/processed/
        │
        ▼  train        para cada modelo em params.yaml:
        │               Pipeline(feature engineering → encoding/scaling → estimador)
        │               → MLflow Tracking (params, 6 métricas, modelo com signature)
        │               → models/model.pkl (o melhor por average_precision)
        ▼  evaluate     promove o melhor run → Model Registry, alias @champion
metrics.json            métricas + resultado da promoção (lido por `dvc metrics show`)
        │
        ▼
FastAPI /predict        carrega models:/online_shoppers_intention@champion
```

O ponto central do desenho: **o pré-processamento faz parte do modelo**, não do estágio de dados.
O artefato registrado recebe as features cruas — as mesmas que a API expõe no payload — então não
existe risco de treino e serving aplicarem transformações diferentes.

## Quick Start

```bash
poetry install
cp .env.example .env
```

### 1. Obter o dataset

```bash
# Dataset real (UCI, 12.330 sessões, ~15,5% de compras) — use este para resultados
python scripts/download_dataset.py

# Ou dado sintético para smoke test do pipeline, sem rede (5.000 sessões)
python scripts/generate_sample_data.py
```

O `.dvc` versionado aponta para o **dataset real** (12.330 sessões, 15,47% de compras), que é a
origem das métricas abaixo. O dado sintético existe para o smoke test da CI, que roda sem rede.
Ao trocar o arquivo, rode `dvc add data/raw/online_shoppers_intention.csv` e commite o ponteiro.

### 2. Rodar o pipeline

```bash
poetry run dvc repro
poetry run dvc metrics show
```

### 3. Ver os experimentos

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://localhost:5000
```

### 4. Subir a API

```bash
poetry run uvicorn ecommerce_buy_predictor.api.main:app --reload
```

```bash
curl http://localhost:8000/health
```

```json
{"status":"healthy","model_loaded":true,"model_source":"models:/online_shoppers_intention@champion"}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 2, "Administrative_Duration": 80.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 31, "ProductRelated_Duration": 1200.5,
    "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 12.4, "SpecialDay": 0.0,
    "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1, "TrafficType": 3,
    "VisitorType": "Returning_Visitor", "Weekend": false
  }'
```

```json
{"prediction":1,"probability":0.87}
```

Documentação interativa em http://localhost:8000/docs. Há também `POST /predict/batch`
para até 1.000 sessões por chamada.

## Docker

```bash
docker build -t ecommerce-buy-predictor .
docker run -p 8000:8000 ecommerce-buy-predictor
```

A imagem embute `models/model.pkl`, então a API responde mesmo sem MLflow no ar. Com o stack
completo, ela prefere o modelo do Registry e informa a origem em `/health`:

```bash
docker compose up --build
# API: http://localhost:8000/docs   MLflow: http://localhost:5000
```

Para popular o Registry do servidor que subiu no compose:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 poetry run dvc repro --force
```

## Estrutura

```
src/ecommerce_buy_predictor/
├── config.py                     # Settings via .env (pydantic-settings)
├── data/
│   ├── loader.py                 # Leitura do CSV
│   └── preprocess.py             # split estratificado + ColumnTransformer
├── features/
│   ├── schema.py                 # fonte única das colunas e dtypes
│   └── build_features.py         # features derivadas da sessão
├── models/
│   ├── train.py                  # Pipeline(features → preprocess → estimador)
│   ├── evaluate.py               # métricas (inclui PR-AUC)
│   └── registry.py               # promoção para o Model Registry
├── pipeline/
│   ├── params.py                 # leitura de params.yaml
│   ├── preprocess_stage.py       # estágio DVC 1
│   ├── train_stage.py            # estágio DVC 2
│   └── evaluate_stage.py         # estágio DVC 3
└── api/
    ├── main.py                   # FastAPI: /health, /predict, /predict/batch
    └── schemas.py                # payload com as 17 features nomeadas
```

## Hiperparâmetros

Tudo em `params.yaml`, que é dependência declarada dos estágios no `dvc.yaml` — mudar um valor
invalida o cache e força o re-treino:

```yaml
selection_metric: average_precision   # métrica que escolhe o modelo promovido
models:
  LogisticRegression:
    max_iter: 1000
  RandomForest:
    n_estimators: 100
    max_depth: 10
```

Adicionar um terceiro modelo é acrescentar uma entrada em `models:` e registrar a classe em
`ESTIMATORS` (`src/ecommerce_buy_predictor/models/train.py`).

## Métricas

A classe positiva é ~15% do total, então **accuracy não é a métrica de decisão**: um modelo que
sempre responde "não compra" acerta 85% e é inútil. O pipeline registra accuracy, precision,
recall, F1, ROC-AUC e average precision (PR-AUC), e usa PR-AUC para escolher o modelo promovido.
Ambos os estimadores usam `class_weight="balanced"`.

Dataset real da UCI (12.330 sessões, 15,47% positivos, split 80/20 com seed 42):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **RandomForest** (promovido) | 0.858 | 0.528 | 0.825 | 0.644 | 0.909 | **0.684** |
| LogisticRegression | 0.841 | 0.492 | 0.743 | 0.592 | 0.893 | 0.622 |

PR-AUC de 0.684 contra uma linha de base de 0.155 (a taxa de positivos): 4,4x melhor que o acaso.
Vale reparar na accuracy: 0.858 é praticamente o que um modelo que responde "não compra" para todo
mundo obteria (0.845). É a diferença entre recall 0.825 e recall 0.0 que importa, e accuracy não
enxerga essa diferença — por isso a seleção usa PR-AUC.

Com o dado sintético do smoke test (5.000 sessões, 14,8% positivos), o promovido é a
LogisticRegression com PR-AUC 0.347 sobre base 0.148.

## Desenvolvimento

```bash
poetry run pytest                       # 37 testes
poetry run pytest --cov=src             # cobertura (97%)
poetry run ruff check src/ tests/ scripts/
```

A CI (`.github/workflows/ci.yml`) roda lint, testes, o pipeline completo (falhando se o modelo não
for promovido) e o build da imagem Docker com um smoke test no `/health`.

## Dados e DVC

`.dvc/` e `dvc.lock` **são versionados** — sem eles não há como reproduzir o pipeline a partir de
um clone. O dataset em si não está no Git; para compartilhá-lo entre máquinas, configure um remote:

```bash
dvc remote add -d storage s3://seu-bucket/tc2   # ou gdrive://, azure://, uma pasta de rede...
dvc push
```

Sem remote configurado, qualquer pessoa reconstrói o dado com `scripts/download_dataset.py`
(real) ou `scripts/generate_sample_data.py` (sintético, determinístico com seed 42).

## Referências

- [Online Shoppers Purchasing Intention Dataset (UCI)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [DVC](https://dvc.org/doc) · [MLflow](https://mlflow.org/docs/) · [FastAPI](https://fastapi.tiangolo.com/) · [Poetry](https://python-poetry.org/docs/)

## Time

Tech Challenge Fase 2 — POSTECH 10MLET.

## Licença

MIT
