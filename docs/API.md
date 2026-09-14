# API de Inferência — Classificação de Laudos Médicos

API HTTP desenvolvida com FastAPI para classificar textos de laudos médicos
por especialidade, utilizando um modelo treinado com TF-IDF e regressão
logística e executado em produção com ONNX Runtime.

Dado o texto de um laudo, a API retorna a categoria prevista pelo modelo.

- **URL pública:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app
- **Execução local:** http://localhost:8000
- **Swagger:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/docs

## Modelo utilizado

O modelo é treinado utilizando:

- `TfidfVectorizer`;
- `LogisticRegression`;
- Pipeline do scikit-learn;
- Exportação para ONNX;
- Inferência com ONNX Runtime.

O pipeline original é salvo em formato Joblib:

```text
artifacts/medical_abstracts_model.joblib
```

Após o treinamento, o modelo é exportado para ONNX:

```text
artifacts/medical_abstracts_model.onnx
```

O modelo utilizado pela API em produção é o artefato ONNX publicado no Cloud Storage:

```text
gs://quantum-balm-260822-medical-models/models/medical_abstracts_model_*.onnx
```

A API carrega o modelo durante a inicialização e utiliza a variável:

```text
MODEL_FORMAT=onnx
```

As categorias previstas são:

```text
- `neoplasms`;
- `digestive`;
- `nervous`;
- `cardiovascular`;
- `general`.
```

## Endpoints

| Método | Rota | Parâmetros | Descrição |
|---|---|---|---|
| `GET` | `/health` | — | Verifica se a API está disponível |
| `POST` | `/predict` | `texto` no corpo JSON | Classifica o texto do laudo |
| `GET` | `/metrics` | — | Expõe métricas no formato Prometheus |
| `GET` | `/docs` | — | Documentação interativa Swagger |
| `GET` | `/openapi.json` | — | Esquema OpenAPI da API |

## Como rodar localmente

Instale as dependências:

```bash
uv sync
```

Prepare os dados, caso necessário:

```bash
uv run python scripts/prepare_dataset_medical_abstracts.py
```

Treine o modelo Joblib:

```bash
uv run python model/train.py
```

Exporte o modelo para ONNX:

```bash
uv run python scripts/export_model_onnx.py
```

Inicie a API utilizando ONNX Runtime:

```bash
MODEL_FORMAT=onnx \
uv run uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Acesse a documentação local:

```text
http://localhost:8000/docs
```

Teste a classificação:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Paciente apresenta alterações no sistema cardiovascular."
  }'
```

Resposta esperada:

```json
{
  "classificacao": "cardiovascular"
}
```

## Como executar com Docker

Construa a imagem:

```bash
docker build \
  -t automated-text-exam-triage:local \
  .
```

Execute o container utilizando ONNX Runtime:

```bash
docker run --rm \
  --name automated-text-exam-triage-api \
  -e MODEL_FORMAT=onnx \
  -p 8000:8080 \
  automated-text-exam-triage:local
```

Acesse a documentação:

```text
http://localhost:8000/docs
```

Teste a API:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Paciente apresenta alterações no sistema cardiovascular."
  }'
```

## Deploy no Google Cloud Run

A API é publicada no Google Cloud Run utilizando uma imagem Docker armazenada
no Artifact Registry.

Em produção, o Cloud Run utiliza o modelo ONNX publicado no Cloud Storage:

```text
Cloud Storage
    │
    ▼
Modelo ONNX versionado
    │
    ▼
Cloud Run
    │
    ▼
FastAPI + ONNX Runtime
```

A configuração atual utiliza:

```text
MODEL_FORMAT=onnx
MODEL_URI=gs://quantum-balm-260822-medical-models/models/medical_abstracts_model_*.onnx
```

A aplicação está disponível em:

- **API:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app
- **Swagger:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/docs
- **Health check:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/health
- **Métricas:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/metrics

Para obter a URL atual do serviço:

```bash
gcloud run services describe medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --format="value(status.url)"
```

Para consultar os logs:

```bash
gcloud run services logs read medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --limit 50
```

## Otimização de inferência

O modelo é treinado originalmente com scikit-learn utilizando TF-IDF e regressão logística.

O pipeline treinado é salvo em formato Joblib:

```text
artifacts/medical_abstracts_model.joblib
```

Depois, o pipeline é exportado para ONNX:

```text
artifacts/medical_abstracts_model.onnx
```

A API em produção utiliza ONNX Runtime para executar as inferências.

O modelo Joblib permanece como:

- artefato intermediário do treinamento;
- referência para validação;
- fallback;
- origem da exportação para ONNX.

### Exportar o modelo

```bash
uv run python scripts/export_model_onnx.py
```

### Comparar os modelos

```bash
uv run python scripts/compare_model_latency.py \
  --warmup 10 \
  --requests 100
```

A comparação valida se o Joblib e o ONNX retornam a mesma classificação antes
de medir a latência.

### Benchmark local de inferência

| Modelo | Requisições | Mínimo (ms) | Média (ms) | Mediana (ms) | P95 (ms) | Máximo (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Joblib/scikit-learn | 100 | 3,305 | 3,427 | 3,422 | 3,557 | 3,624 |
| ONNX Runtime | 100 | 1,908 | 2,114 | 2,133 | 2,258 | 2,315 |

Considerando a latência média:

- Joblib/scikit-learn: **3,427 ms**;
- ONNX Runtime: **2,114 ms**;
- Redução média: **38,31%**;
- Ganho de velocidade: aproximadamente **1,62x**.

### Benchmark no Cloud Run

Foram realizadas 100 requisições ao endpoint `/predict`, após 10 requisições
de aquecimento. Todas as requisições retornaram `HTTP 200`.

| Métrica | Valor |
|---|---:|
| Requisições | 100 |
| Aquecimento | 10 |
| Latência mínima | 166,885 ms |
| Latência média | 173,961 ms |
| Mediana | 171,603 ms |
| P95 | 183,357 ms |
| Latência máxima | 312,267 ms |

O resultado foi salvo em:

```text
benchmarks/latency-cloud-run-onnx.json
```

A latência do Cloud Run inclui a comunicação HTTP, rede, FastAPI, middleware, serialização da resposta, infraestrutura do Cloud Run e inferência com ONNX Runtime. Por isso, ela não deve ser comparada diretamente com a latência isolada da inferência local.

## Orquestração com Airflow

O Airflow executa mensalmente o fluxo de ingestão, treinamento, exportação e
deploy do modelo:

```text
ingest_data
    ↓
train_model
    ↓
export_model_onnx
    ↓
publish_model
    ↓
Cloud Run
```

A DAG:

1. Baixa e prepara os dados;
2. Treina o modelo Joblib;
3. Exporta o modelo para ONNX;
4. Publica o modelo ONNX no Cloud Storage;
5. Atualiza a variável `MODEL_URI`;
6. Configura `MODEL_FORMAT=onnx`;
7. Cria uma nova revisão no Cloud Run.

A execução ocorre no primeiro dia de cada mês, às 2h, no fuso
`America/Sao_Paulo`.

Para iniciar o Airflow:

```bash
docker compose up --build -d airflow
```

A interface está disponível em:

```text
http://localhost:8080
```

Mais informações:

[Documentação completa do Airflow](docs/AIRFLOW.md)

## Observações

- A API pública utiliza o Google Cloud Run;
- O modelo de produção é executado com ONNX Runtime;
- O modelo Joblib é utilizado como artefato intermediário e fallback;
- O modelo ONNX é versionado no Cloud Storage;
- O Airflow executa mensalmente o retreinamento e o deploy do modelo;
- A primeira requisição pode apresentar maior latência devido ao cold start;
- O modelo é carregado em memória durante a inicialização da aplicação;
- O endpoint `/predict` recebe o texto no corpo JSON;
- Prometheus e Grafana são utilizados para monitoramento local;
- O Cloud Run utiliza métricas nativas do Google Cloud;
- O modelo atual classifica especialidades médicas, não níveis de urgência.
