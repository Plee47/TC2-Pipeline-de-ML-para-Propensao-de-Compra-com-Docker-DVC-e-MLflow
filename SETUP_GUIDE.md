# Setup Guide — Tech Challenge Fase 2

## ✅ Concluído (Etapas 0-1)

### Etapa 0 — Bootstrap
- [x] Repositório Git inicializado
- [x] `.gitignore` e `.dockerignore` configurados
- [x] Estrutura de pastas criada (src/, tests/, data/, models/, etc.)
- [x] Primeiro commit com projeto base

### Etapa 1 — Clean Code e Estrutura
- [x] `pyproject.toml` configurado com Poetry
- [x] Dependências separadas (prod/dev)
- [x] Módulos Python com type hints:
  - `ecommerce_buy_predictor/config.py` — leitura de `.env`
  - `ecommerce_buy_predictor/data/loader.py` — carregar CSV
  - `ecommerce_buy_predictor/data/preprocess.py` — `Preprocessor` class
  - `ecommerce_buy_predictor/models/train.py` — `ModelTrainer` class (LogReg + RF)
  - `ecommerce_buy_predictor/models/evaluate.py` — métricas e logging MLflow
  - `ecommerce_buy_predictor/models/registry.py` — promoção ao Model Registry
  - `ecommerce_buy_predictor/pipeline/*.py` — 3 estágios DVC
  - `ecommerce_buy_predictor/api/main.py` — FastAPI app com `/health` e `/predict`
- [x] `.env.example` com placeholders
- [x] `.env` criado localmente
- [x] Testes unitários (`tests/`)
- [x] README.md completo com Quick Start
- [x] `dvc.yaml` com pipeline
- [x] `params.yaml` com hiperparâmetros
- [x] `Dockerfile` e `docker-compose.yml`
- [x] Dataset de exemplo gerado (5000 amostras com feature engineering)

## 📦 Próximos passos (Etapas 2-4)

### Etapa 2 — Ambiente e Dependências

Aguardando instalação de Poetry e DVC em background. Assim que completarem:

```bash
cd "C:\Users\Aline\OneDrive\Desktop\MBA\mba.fiap.ecommerce.buy.predictor"

# 1. Instalar dependências do projeto
poetry install

# 2. Gerar lock file (já incluído em pyproject.toml)
poetry lock

# 3. Validar instalação
poetry show
```

### Etapa 3 — Containerização e Versionamento

Inicializar DVC:

```bash
dvc init
dvc remote add -d local ../dvc-storage
dvc add data/raw/online_shoppers_intention.csv
dvc commit
```

Rodar pipeline DVC:

```bash
dvc repro
```

Visualizar pipeline:

```bash
dvc dag
```

Testar Dockerfile:

```bash
docker build -t tech-challenge .
docker run -p 8000:8000 tech-challenge
```

### Etapa 4 — Modelagem, Registry e Entrega

Após `dvc repro` completar com sucesso:

1. Visualizar experimentos no MLflow:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Acessa: http://localhost:5000

2. Verificar se modelo foi promovido ao Registry:
   ```bash
   mlflow models list
   ```

3. Testar API:
   ```bash
   poetry run python -m uvicorn ecommerce_buy_predictor.api.main:app --reload
   curl http://localhost:8000/health
   ```

4. Rodar testes:
   ```bash
   poetry run pytest -v
   ```

5. Validar com Ruff:
   ```bash
   poetry run ruff check src/
   ```

## 📋 Instalação de ferramentas (em background)

Status de instalações acionadas:
- [ ] DVC (em progresso)
- [ ] Poetry (em progresso)

Se não completarem automaticamente, instale manualmente:

```bash
python -m pip install poetry dvc mlflow scikit-learn pandas fastapi uvicorn pydantic-settings pytest ruff httpx -q
```

## 🔑 Configurações necessárias

Editar `.env` com credenciais Kaggle (opcional):
```
KAGGLE_USERNAME=seu_username
KAGGLE_KEY=sua_chave_api
```

Se quiser baixar o dataset real do Kaggle:
```bash
kaggle datasets download -d sagarshrivastava/online-shoppers-intention-to-purchase -p data/raw/
```

Senão, o script `scripts/generate_sample_data.py` já gerou um dataset de exemplo.

## 📁 Estrutura final esperada

```
mba.fiap.ecommerce.buy.predictor/
├── src/ecommerce_buy_predictor/
│   ├── __init__.py
│   ├── config.py
│   ├── data/ (loader.py, preprocess.py)
│   ├── features/ (build_features.py)
│   ├── models/ (train.py, evaluate.py, registry.py)
│   ├── pipeline/ (preprocess_stage.py, train_stage.py, evaluate_stage.py)
│   └── api/ (main.py, schemas.py)
├── tests/ (test_*.py)
├── data/
│   ├── raw/ (online_shoppers_intention.csv + .dvc)
│   └── processed/ (X_train, X_test, y_train, y_test + preprocessor)
├── models/ (artefatos locais)
├── scripts/ (generate_sample_data.py)
├── .dvc/ (versionamento)
├── mlruns/ (MLflow tracking)
├── pyproject.toml
├── poetry.lock
├── .env
├── .env.example
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .git/ (histórico)
```

## ✨ Clean Code Checklist

- ✅ Funções ≤ 20 linhas
- ✅ Type hints em tudo
- ✅ Naming descritivo
- ✅ POO leve (Preprocessor, ModelTrainer)
- ✅ Sem duplicação
- ✅ Testes unitários
- ✅ Docstrings mínimas (apenas quando necessário)

## 🎯 Rubric de Avaliação

| Critério | Peso | Status |
|----------|------|--------|
| Clean Code e Estrutura | 20% | ✅ Pronto |
| Reprodutibilidade | 20% | 🔄 Aguardando Poetry |
| Docker | 15% | ✅ Dockerfile criado |
| DVC + Pipeline | 15% | 🔄 Aguardando DVC |
| Modelagem Clássica | 10% | 🔄 Código pronto, aguardando execução |
| MLflow + Registry | 20% | 🔄 Código pronto, aguardando execução |

## 🚀 Próxima ação

Aguardar conclusão de:
1. Instalação de Poetry
2. Instalação de DVC
3. Executar `poetry install`
4. Executar `dvc repro`

Depois proceder com testes e validação.
