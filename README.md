# Tech Challenge Fase 2 — Online Shoppers Purchasing Intention Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-managed-brightgreen.svg)](https://python-poetry.org/)
[![DVC](https://img.shields.io/badge/dvc-versioned-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/mlflow-tracked-red.svg)](https://mlflow.org/)

Uma solução de **Engenharia de Machine Learning** para prever a propensão de compra de usuários em e-commerce,
com foco em um pipeline containerizado, reprodutível e profissional.

## 🎯 Objetivo

Desenvolver um sistema preditivo que identifique a propensão de compra de um usuário baseado em seu comportamento de navegação,
usando boas práticas de Clean Code, versionamento de dados (DVC), rastreamento de experimentos (MLflow)
e containerização (Docker).

## 📋 Requisitos

- Python 3.10+
- Poetry (gerenciador de dependências)
- Docker & Docker Compose (opcional, para containerização)
- Kaggle API (opcional, para download automático do dataset)

## 🚀 Quick Start

### 1. Preparação do Ambiente

```bash
# Clonar repositório
cd mba.fiap.ecommerce.buy.predictor

# Instalar dependências
poetry install

# Copiar .env.example para .env e preencher credenciais
cp .env.example .env
# Editar .env com suas configurações
```

### 2. Baixar Dataset

#### Opção A: Via Kaggle API (recomendado)

```bash
# Configurar Kaggle (se ainda não fez)
# https://kaggle.com/settings/account → Create New Token
# Vai criar ~/.kaggle/kaggle.json

# Baixar dataset
kaggle datasets download -d sagarshrivastava/online-shoppers-intention-to-purchase -p data/raw/
unzip data/raw/online-shoppers-intention-to-purchase.zip -d data/raw/
rm data/raw/online-shoppers-intention-to-purchase.zip
```

#### Opção B: Download Manual

Baixar manualmente em https://kaggle.com/datasets/sagarshrivastava/online-shoppers-intention-to-purchase
e salvar `online_shoppers_intention.csv` em `data/raw/`.

### 3. Inicializar DVC e Pipeline

```bash
# Inicializar DVC (se não feito)
dvc init

# Executar pipeline completo (preprocess → train → evaluate)
dvc repro

# Visualizar métricas
dvc metrics show
```

### 4. Visualizar Experimentos no MLflow

```bash
# Abrir MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Acessa http://localhost:5000 para ver:
# - Runs de LogisticRegression e RandomForest
# - Métricas (accuracy, precision, recall, F1, ROC-AUC)
# - Modelos registrados no Registry
```

### 5. Rodar API Localmente

```bash
# Via Poetry
poetry run python -m uvicorn ecommerce_buy_predictor.api.main:app --reload --host 0.0.0.0 --port 8000

# Ou via Docker
docker build -t ecommerce-predictor .
docker run -p 8000:8000 ecommerce-predictor
```

Testar endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Fazer predição (exemplo)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

## 📁 Estrutura do Projeto

```
mba.fiap.ecommerce.buy.predictor/
├── src/ecommerce_buy_predictor/
│   ├── config.py                 # Configurações via .env
│   ├── data/
│   │   ├── loader.py             # Carregar CSV
│   │   └── preprocess.py         # Preprocessamento
│   ├── features/
│   │   └── build_features.py     # Feature engineering
│   ├── models/
│   │   ├── train.py              # Treinamento (LogReg, RandomForest)
│   │   ├── evaluate.py           # Avaliação e métricas
│   │   └── registry.py           # MLflow Model Registry
│   ├── pipeline/
│   │   ├── preprocess_stage.py   # DVC stage: preprocess
│   │   ├── train_stage.py        # DVC stage: train
│   │   └── evaluate_stage.py     # DVC stage: evaluate
│   └── api/
│       ├── main.py               # FastAPI app
│       └── schemas.py            # Pydantic schemas
├── tests/                        # Testes unitários
├── data/
│   ├── raw/                      # Dataset original (versionado DVC)
│   └── processed/                # Dados processados (treino/teste)
├── models/                       # Modelos locais
├── pyproject.toml                # Dependências Poetry
├── dvc.yaml                      # Pipeline DVC
├── params.yaml                   # Hiperparâmetros
├── Dockerfile                    # Containerização
├── docker-compose.yml            # Orquestração local
└── README.md                     # Este arquivo
```

## 🧪 Testes

```bash
# Rodar testes
poetry run pytest -v

# Com cobertura
poetry run pytest --cov=src tests/
```

## 🔍 Qualidade de Código

```bash
# Lint com Ruff
poetry run ruff check src/

# Fix automático
poetry run ruff check --fix src/
```

## 🐳 Containerização (Docker)

### Compilar imagem

```bash
docker build -t tech-challenge:latest .
```

### Rodar container

```bash
docker run -p 8000:8000 tech-challenge:latest
```

### Subir stack completo (MLflow + API)

```bash
docker-compose up
```

Acessa:
- API: http://localhost:8000/docs (Swagger UI)
- MLflow: http://localhost:5000

## 🔄 Pipeline DVC

O pipeline está definido em `dvc.yaml` com 3 estágios:

1. **preprocess**: Carrega dados brutos, aplica limpeza, encoding e split (80/20)
2. **train**: Treina 2 modelos (LogisticRegression + RandomForest) com MLflow tracking
3. **evaluate**: Promove melhor modelo para MLflow Model Registry (Production)

Rodar pipeline:

```bash
dvc repro
```

Visualizar dependências:

```bash
dvc dag
```

## 📊 Modelos

### LogisticRegression
- Interpretável, rápido
- Útil para baseline

### RandomForest
- Captura não-linearidades
- Robusto a dados desbalanceados

Ambos são rastreados no MLflow Tracking com:
- Parâmetros (n_estimators, max_iter, etc.)
- Métricas (accuracy, precision, recall, F1, ROC-AUC)
- Artefatos (modelo pkl, exemplos de entrada)

Melhor modelo é promovido para `Production` no Model Registry.

## 🌐 Deploy (Opcional)

### Em Render.com

```bash
# Criar Render service com imagem Docker
# - URL: https://github.com/<seu-repo>/mba.fiap.ecommerce.buy.predictor
# - Dockerfile presente
# - Portar 8000

# Acessar
curl https://<seu-render-app>.onrender.com/health
```

### Em AWS App Runner

```bash
aws apprunner create-service \
  --service-name tech-challenge \
  --source-configuration RepositoryType=GITHUB,ImageRepository=...
```

### Em Cloud Run (GCP)

```bash
gcloud run deploy tech-challenge \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📝 Clean Code Principles

- ✅ Funções curtas (≤20 linhas)
- ✅ Type hints em tudo
- ✅ Naming conventions descritivas
- ✅ POO leve (Preprocessor, ModelTrainer)
- ✅ Sem duplicação de código
- ✅ Testes unitários

## 🔗 Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/sagarshrivastava/online-shoppers-intention-to-purchase)
- [Poetry Docs](https://python-poetry.org/docs/)
- [DVC Docs](https://dvc.org/doc)
- [MLflow Docs](https://mlflow.org/docs/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 👥 Time

Desenvolvido como Tech Challenge Fase 2 — POSTECH — 10MLET

## 📄 Licença

MIT
