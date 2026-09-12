# Airflow — Ingestão e Retreinamento

O Airflow orquestra mensalmente o processo de ingestão dos dados, retreinamento
do modelo e atualização da API no Google Cloud Run.

> O modelo atual classifica especialidades médicas.

## Fluxo da DAG

```text
ingest_data → train_model → publish_model
```

### `ingest_data`

A task reutiliza o script:

```text
scripts/prepare_dataset_medical_abstracts.py
```

Essa etapa:

1. Baixa a versão mais recente do dataset pelo Kaggle;
2. Localiza os arquivos CSV;
3. Combina os dados;
4. Padroniza as colunas;
5. Remove registros inválidos;
6. Salva o dataset preparado em:

```text
data/laudos.csv
```

### `train_model`

A task reutiliza o script:

```text
model/train.py
```

Essa etapa:

1. Carrega `data/laudos.csv`;
2. Divide os dados em treino e teste;
3. Treina o pipeline TF-IDF + regressão logística;
4. Avalia o modelo;
5. Salva o artefato em:

```text
artifacts/medical_abstracts_model.joblib
```

### `train_model`

Essa task:

1. Envia o modelo treinado para o Cloud Storage;
2. Cria uma versão identificável do artefato;
3. Atualiza a variável MODEL_URI do Cloud Run;
4. Cria uma nova revisão do serviço;
5. Direciona o tráfego para a nova revisão.

O modelo é publicado em um caminho semelhante a:

```text
gs://quantum-balm-260822-medical-models/models/medical_abstracts_model_20260912003343.joblib
```

## Agendamento

A DAG é executada no primeiro dia de cada mês, às 2h, no fuso `America/Sao_Paulo`:

```python
schedule="0 2 1 * *"
```

A DAG utiliza:

```python
catchup=False
```

Assim, o Airflow não executa automaticamente os períodos anteriores que estiverem pendentes.

## Estrutura relacionada ao Airflow

```text
airflow/
├── dags/
│   └── train_model_dag.py
├── logs/
├── plugins/
├── secrets/
├── Dockerfile
└── requirements.txt

scripts/
└── prepare_dataset_medical_abstracts.py

model/
└── train.py

data/
└── laudos.csv

artifacts/
└── medical_abstracts_model.joblib
```

## Pré-requisitos

- Docker;
- Docker Compose;
- Projeto GCP com faturamento habilitado;
- Google Cloud CLI;
- Bucket no Cloud Storage;
- Serviço publicado no Cloud Run;
- Conta de serviço para o Airflow.

## Configuração do GCP

O projeto utiliza:

```text
Projeto: quantum-balm-260822
Região: us-central1
Serviço Cloud Run: medical-exam-api
Bucket: quantum-balm-260822-medical-models
Repositório Artifact Registry: medical-exam
```

A conta de serviço utilizada pelo Airflow é:

```text
airflow-deployer@quantum-balm-260822.iam.gserviceaccount.com
```

Ela precisa possuir as permissões:

```text
roles/storage.objectAdmin
roles/run.admin
roles/artifactregistry.reader
```

A conta de execução do Cloud Run precisa possuir permissão para ler o modelo:

```text
roles/storage.objectViewer
```

## Credenciais do Airflow

A chave da conta de serviço deve estar no arquivo:

```text
secrets/airflow-deployer.json
```

Ajuste as permissões:

```bash
chmod 600 secrets/airflow-deployer.json
```

O arquivo deve ser montado no container em:

```text
/opt/airflow/secrets/airflow-deployer.json
```

## Iniciar o Airflow

Na raiz do projeto, execute:

```bash
docker compose up --build -d airflow
```

Verifique o container:

```bash
docker compose ps airflow
```

A interface web estará disponível em:

```text
http://localhost:8080
```

## Acessar a interface

Abra:

```text
http://localhost:8080
```

O usuário padrão é:

```text
admin
```

Para consultar a senha gerada automaticamente:

```bash
docker compose logs airflow | grep -i -A 5 "password"
```

Na interface do Airflow:

1. Localize a DAG `train_medical_abstracts_model`;
2. Ative a DAG, caso esteja pausada;
3. Consulte o próximo agendamento;
4. Acompanhe as tasks `ingest_data`, `train_model` e `publish_model`.

## Verificar a DAG pelo terminal

Listar as DAGs:

```bash
docker compose exec airflow airflow dags list
```

Verificar erros de importação:

```bash
docker compose exec airflow \
  airflow dags list-import-errors
```

Ver detalhes da DAG:

```bash
docker compose exec airflow \
  airflow dags details train_medical_abstracts_model
```

Listar as execuções:

```bash
docker compose exec airflow \
  airflow dags list-runs \
  --dag-id train_medical_abstracts_model
```

## Autenticar o Google Cloud CLI

O container do Airflow utiliza a conta de serviço para acessar o Cloud Storage,
o Artifact Registry e o Cloud Run.

Se necessário, autentique o `gcloud` dentro do container:

```bash
docker compose exec airflow \
  gcloud auth activate-service-account \
  airflow-deployer@quantum-balm-260822.iam.gserviceaccount.com \
  --key-file=/opt/airflow/secrets/airflow-deployer.json \
  --project=quantum-balm-260822
```

Confirme a conta ativa:

```bash
docker compose exec airflow gcloud auth list
```

O resultado esperado é:

```text
ACTIVE  ACCOUNT
*       airflow-deployer@quantum-balm-260822.iam.gserviceaccount.com
```

Confirme o projeto:

```bash
docker compose exec airflow \
  gcloud config get-value project
```

Resultado esperado:

```text
quantum-balm-260822
```

## Testar o acesso ao Cloud Storage

```bash
docker compose exec airflow \
  gcloud storage ls \
  gs://quantum-balm-260822-medical-models
```

Listar os modelos publicados:

```bash
gcloud storage ls \
  gs://quantum-balm-260822-medical-models/models/
```

## Testar o acesso ao Artifact Registry

```bash
docker compose exec airflow \
  gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/quantum-balm-260822/medical-exam \
  --include-tags
```

A imagem da API deve aparecer na saída.

## Testar a atualização do Cloud Run

Antes de executar a DAG, é possível validar se o Airflow consegue atualizar
o serviço:

```bash
docker compose exec airflow \
  gcloud run services update medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --update-env-vars="TEST_DEPLOY=true"
```

Se o comando funcionar, remova a variável de teste:

```bash
docker compose exec airflow \
  gcloud run services update medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --remove-env-vars="TEST_DEPLOY"
```

## Executar a DAG manualmente

Para testar a DAG sem aguardar o agendamento mensal:

```bash
docker compose exec airflow \
  airflow dags test \
  train_medical_abstracts_model \
  2026-09-15
```

A execução esperada é:

```text
ingest_data      SUCCESS
train_model      SUCCESS
publish_model    SUCCESS
```

Para disparar uma execução normal:

```bash
docker compose exec airflow \
  airflow dags trigger train_medical_abstracts_model
```

Acompanhar os logs:

```bash
docker compose logs -f airflow
```

## Validar os artefatos

Depois de uma execução bem-sucedida:

```bash
ls -lh data/laudos.csv
ls -lh artifacts/medical_abstracts_model.joblib
```

Verifique o modelo publicado no Cloud Storage:

```bash
gcloud storage ls \
  gs://quantum-balm-260822-medical-models/models/
```

## Verificar o modelo utilizado pelo Cloud Run

Consulte a variável `MODEL_URI`:

```bash
gcloud run services describe medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --format="yaml(spec.template.spec.containers[0].env)"
```

A saída deve conter algo semelhante a:

```yaml
- name: MODEL_URI
  value: gs://quantum-balm-260822-medical-models/models/medical_abstracts_model_20260912003343.joblib
```

## Verificar as revisões do Cloud Run

```bash
gcloud run revisions list \
  --service=medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1
```

A revisão mais recente deve estar ativa e recebendo 100% do tráfego.

Para consultar a distribuição de tráfego:

```bash
gcloud run services describe medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --format="yaml(status.traffic)"
```

## Testar a API após o deploy

A URL pública da API pode ser obtida com:

```bash
gcloud run services describe medical-exam-api \
  --project=quantum-balm-260822 \
  --region=us-central1 \
  --format="value(status.url)"
```

Health check:

```bash
curl -s \
  https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/health
```

Classificação:

```bash
curl -s -X POST \
  https://medical-exam-api-ppsmoa2pqq-uc.a.run.app/predict \
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

## Reiniciar o Airflow

Após alterar a DAG:

```bash
docker compose restart airflow
```

Se houver alteração no `Dockerfile` ou no `airflow/requirements.txt`:

```bash
docker compose down
docker compose build --no-cache airflow
docker compose up -d airflow
```

## Volumes utilizados

O serviço do Airflow monta os diretórios do projeto no container:

```yaml
volumes:
  - ./airflow/dags:/opt/airflow/dags
  - ./airflow/logs:/opt/airflow/logs
  - ./airflow/plugins:/opt/airflow/plugins
  - ./airflow/secrets:/opt/airflow/secrets:ro
  - ./scripts:/opt/airflow/scripts
  - ./model:/opt/airflow/model
  - ./data:/opt/airflow/data
  - ./artifacts:/opt/airflow/artifacts
```

Esses volumes permitem que:

- o Airflow acesse a DAG;
- a DAG reutilize os scripts existentes;
- o dataset seja salvo em `data/`;
- o modelo seja salvo em `artifacts/`;
- a chave da conta de serviço seja acessada com segurança;
- os arquivos gerados sejam acessíveis no host.

## Permissões

O Airflow executa como o usuário `airflow` dentro do container. Os diretórios
de logs, dados e artefatos precisam permitir escrita:

```bash
sudo chown -R 50000:0 \
  airflow/logs \
  airflow/plugins \
  data \
  artifacts

sudo chmod -R ug+rwX \
  airflow/logs \
  airflow/plugins \
  data \
  artifacts
```

A chave da conta de serviço deve permanecer protegida:

```bash
chmod 600 secrets/airflow-deployer.json
```

## Limpeza do ambiente

Para parar os serviços:

```bash
docker compose down
```

Para remover também os volumes do Docker:

```bash
docker compose down -v
```

> O comando `docker compose down -v` pode remover dados persistidos de outros
> serviços da stack. Utilize-o apenas quando quiser reiniciar o ambiente.

## Fluxo final

```text
Kaggle
  ↓
ingest_data
  ↓
data/laudos.csv
  ↓
train_model
  ↓
artifacts/medical_abstracts_model.joblib
  ↓
Cloud Storage
  ↓
publish_model
  ↓
Nova revisão do Cloud Run
  ↓
API utilizando o modelo atualizado
```

O processo mensal está configurado para realizar automaticamente:

1. Ingestão do dataset;
2. Preparação dos dados;
3. Retreinamento do modelo;
4. Avaliação do modelo;
5. Publicação do artefato no Cloud Storage;
6. Atualização da variável `MODEL_URI`;
7. Criação de uma nova revisão no Cloud Run.
