# API de Inferência — Classificação de Laudos Médicos

API HTTP desenvolvida com FastAPI para classificar textos de laudos médicos usando um modelo treinado com TF-IDF e regressão logística.

Dado o texto de um laudo, a API retorna a categoria prevista pelo modelo.

- **URL pública:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app
- **Execução local:** http://localhost:8000
- **Swagger:** https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/docs

## Modelo utilizado

O modelo utiliza:

- `TfidfVectorizer`;
- `LogisticRegression`;
- Pipeline do scikit-learn;
- Persistência em formato Joblib.

O artefato utilizado pela API é:

```text
artifacts/medical_abstracts_model.joblib
```

As categorias previstas são:

- `neoplasms`;
- `digestive`;
- `nervous`;
- `cardiovascular`;
- `general`.

## Endpoints

| Método | Rota | Parâmetros | Descrição |
|---|---|---|---|
| `GET` | `/health` | — | Verifica se a API está disponível |
| `POST` | `/predict` | `texto` no corpo JSON | Classifica o texto do laudo |
| `GET` | `/metrics` | — | Expõe métricas no formato Prometheus |
| `GET` | `/docs` | — | Documentação interativa Swagger |
| `GET` | `/openapi.json` | — | Esquema OpenAPI da API |

## Classificação de um laudo

O endpoint `/predict` recebe um corpo JSON com o campo obrigatório `texto`.

### Requisição

```bash
curl -X POST \
  https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texto": "Paciente apresenta alterações no sistema cardiovascular."
  }'
```

### Resposta

```json
{
  "classificacao": "cardiovascular"
}
```

A resposta pode conter uma das seguintes categorias:

```text
neoplasms
digestive
nervous
cardiovascular
general
```

## Health check

Para verificar se a API está disponível:

```bash
curl https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/health
```

Resposta esperada:

```json
{
  "status": "ok"
}
```

## Métricas

As métricas da API podem ser consultadas em:

```bash
curl https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/metrics
```

Entre as métricas disponíveis estão:

```text
api_requests_total
api_request_duration_seconds
```

As métricas incluem:

- Quantidade de requisições;
- Método HTTP;
- Endpoint;
- Código de status;
- Tempo de resposta.

## Testar pelo Swagger

1. Acesse:

   https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/docs

2. Expanda o endpoint `POST /predict`;
3. Clique em **Try it out**;
4. Informe o texto do laudo:

```json
{
  "texto": "Paciente apresenta alterações no sistema cardiovascular."
}
```

5. Clique em **Execute**;
6. Consulte o resultado em **Response body**.

## Como rodar localmente

Instale as dependências:

```bash
uv sync
```

Caso o modelo ainda não exista, execute o treinamento:

```bash
uv run python model/train.py
```

Inicie a API:

```bash
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

## Como executar com Docker

Construa a imagem:

```bash
docker build \
  -t automated-text-exam-triage:local \
  .
```

Execute o container:

```bash
docker run --rm \
  --name automated-text-exam-triage-api \
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

A API é publicada no Google Cloud Run utilizando uma imagem Docker armazenada no Artifact Registry.

Arquitetura utilizada:

```text
Dockerfile
    │
    ▼
Artifact Registry
    │
    ▼
Google Cloud Run
    │
    ▼
FastAPI
    │
    ▼
Modelo Joblib
```

Para obter a URL do serviço:

```bash
gcloud run services describe medical-exam-api \
  --region us-central1 \
  --format="value(status.url)"
```

Para consultar os logs:

```bash
gcloud run services logs read medical-exam-api \
  --region us-central1 \
  --limit 50
```

## Benchmark de latência

O projeto possui um script para medir a latência do endpoint `/predict`:

```bash
uv run python scripts/measure_latency.py \
  --url https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/predict \
  --warmup 10 \
  --requests 100 \
  --output artifacts/latency-cloud-run.json
```

O benchmark calcula:

- Latência mínima;
- Latência média;
- Mediana;
- Percentil 95 (`P95`);
- Latência máxima.

## Comparação entre Joblib e ONNX Runtime

O modelo original é salvo em formato Joblib. Também existe uma versão exportada para ONNX Runtime para comparação de latência.

Exporte o modelo para ONNX:

```bash
uv run python scripts/export_model_onnx.py
```

Compare os modelos:

```bash
uv run python scripts/compare_model_latency.py \
  --warmup 10 \
  --requests 100
```

A comparação valida se os dois modelos retornam a mesma classificação antes de medir o tempo de inferência.

Resultado obtido:

| Modelo | Requisições | Mínimo (ms) | Média (ms) | Mediana (ms) | P95 (ms) | Máximo (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Joblib/scikit-learn | 100 | 3,305 | 3,427 | 3,422 | 3,557 | 3,624 |
| ONNX Runtime | 100 | 1,908 | 2,114 | 2,133 | 2,258 | 2,315 |

O modelo ONNX Runtime apresentou redução aproximada de:

- **38,31% na latência média**;
- **36,53% na latência P95**.

## Monitoramento local

A API, o Prometheus, o Grafana e o Airflow podem ser executados juntos com Docker Compose:

```bash
docker compose up --build -d
```

Serviços disponíveis:

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Airflow: http://localhost:8080

Credenciais padrão do Grafana:

```text
Usuário: admin
Senha: admin
```

Para gerar tráfego e popular os gráficos:

```bash
uv run python scripts/generate_traffic.py \
  --url http://localhost:8000/predict \
  --requests 200
```

Para verificar os serviços:

```bash
docker compose ps
```

Para visualizar os logs:

```bash
docker compose logs -f
```

Para encerrar os serviços:

```bash
docker compose down
```

## Monitoramento no Google Cloud

No ambiente local, Prometheus e Grafana são utilizados para visualizar as métricas da API.

No Google Cloud Run, podem ser utilizados:

- Cloud Monitoring;
- Cloud Logging;
- Cloud Trace;
- Managed Service for Prometheus.

Para consultar os logs do serviço:

```bash
gcloud run services logs read medical-exam-api \
  --region us-central1 \
  --limit 50
```

## Observações

- A API pública utiliza o Google Cloud Run;
- A primeira requisição pode apresentar maior latência devido ao cold start;
- O modelo é carregado em memória durante a inicialização da aplicação;
- O endpoint `/predict` recebe o texto no corpo JSON;
- O modelo Joblib é utilizado pela API principal;
- O modelo ONNX Runtime foi utilizado para comparação de latência;
- Prometheus e Grafana são utilizados para monitoramento local;
- O Cloud Run utiliza métricas nativas do Google Cloud;
- O modelo atual classifica categorias médicas, não níveis de urgência.
