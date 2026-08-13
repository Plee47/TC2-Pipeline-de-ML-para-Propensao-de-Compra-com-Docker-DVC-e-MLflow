# Reprodutibilidade

O objetivo aqui é simples: quem clonar este repositório consegue chegar aos mesmos números,
e consegue provar que chegou.

## Pré-requisitos

- Python 3.10–3.12
- Poetry 1.8+ (com Poetry 2.x, `poetry export` exige `poetry-plugin-export`; a imagem Docker já instala os dois)
- Git

DVC e MLflow vêm como dependências do projeto, não precisam ser instalados à parte.

## Reprodução completa

```bash
git clone <repo-url> && cd TC2-Pipeline-de-ML-para-Propensao-de-Compra-com-Docker-DVC-e-MLflow
poetry install
python scripts/download_dataset.py         # ou generate_sample_data.py para o smoke test
poetry run dvc repro
```

Saída esperada do `dvc repro` (dataset real da UCI, seed 42):

```
Running stage 'preprocess':
Preprocessing complete: 9864 train / 2466 test rows saved to data\processed

Running stage 'train':
LogisticRegression: {'accuracy': 0.8414, 'precision': 0.4922, 'recall': 0.7435, 'f1': 0.5923,
                     'roc_auc': 0.8931, 'average_precision': 0.6222}
RandomForest:       {'accuracy': 0.8585, 'precision': 0.5276, 'recall': 0.8246, 'f1': 0.6435,
                     'roc_auc': 0.9094, 'average_precision': 0.6835}
Best model: RandomForest (average_precision=0.6835) saved to models\model.pkl

Running stage 'evaluate':
Promoted models:/online_shoppers_intention@champion (version 1)
```

Com o dado sintético (`generate_sample_data.py`), são 4000/1000 linhas e a LogisticRegression
vence com `average_precision=0.3467`.

Verificação:

```bash
poetry run dvc metrics show     # métricas reais + promoted: true
poetry run dvc status           # "Data and pipelines are up to date."
poetry run pytest               # 37 passed
poetry run ruff check src/ tests/ scripts/
```

## O que garante o determinismo

| Fonte de variação | Como está controlada |
|---|---|
| Split treino/teste | `random_seed: 42` em `params.yaml`, propagado ao `train_test_split` |
| Estimadores | mesmo seed em `LogisticRegression` e `RandomForestClassifier` |
| Dado sintético | `numpy.random.default_rng(42)` em `generate_sample_data` |
| Versões de biblioteca | `poetry.lock` versionado |
| Estado do pipeline | `dvc.lock` versionado — inclui hash dos dados, do código e dos parâmetros |
| Ambiente | `Dockerfile` com dependências exportadas do lock |

Rodar `dvc repro` duas vezes na mesma máquina produz métricas idênticas dígito a dígito. Entre
máquinas diferentes, pode haver variação na última casa decimal por diferenças de BLAS/CPU.

**Não é determinístico, e nem precisa ser:** run_id do MLflow (UUID), timestamps, tempo de execução.

## Dados

`data/raw/online_shoppers_intention.csv` é rastreado por DVC (`.dvc` versionado, arquivo não).
Como o repositório não tem um remote público configurado, o dado é reconstruído localmente:

| Comando | Resultado |
|---|---|
| `python scripts/download_dataset.py` | dataset real da UCI, 12.330 sessões, 15,47% positivos — é o que o `.dvc` versionado referencia |
| `python scripts/generate_sample_data.py` | 5.000 sessões sintéticas, 14,8% positivos, determinístico, sem rede |

Para compartilhar o dado entre máquinas de verdade: `dvc remote add -d storage <url>` e `dvc push`.

## Problemas comuns

**`ERROR: you are not inside of a DVC repository`** — versão antiga do repo, em que `.dvc/` estava
no `.gitignore`. Atualize a branch.

**`The requested command export does not exist` no build Docker** — Poetry 2.x sem o plugin de
export. O Dockerfile já fixa `poetry==1.8.5` + `poetry-plugin-export`.

**`/predict` responde 503** — nenhum modelo carregado. Rode `dvc repro` (que promove o modelo) ou
confira `GET /health`, que informa a origem do modelo em uso.

**Métricas diferentes das da tabela** — quase sempre é dataset diferente (real vs sintético) ou
`params.yaml` alterado. `dvc metrics diff` mostra a diferença entre commits.
