# Airflow — Ingestão e Retreinamento

O Airflow é utilizado para orquestrar o processo mensal de ingestão dos dados
e retreinamento do modelo de classificação de especialidades médicas.

> O modelo atual classifica especialidades médicas, não níveis de urgência.

## Fluxo da DAG

```text
ingest_data → train_model
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

Utilize as credenciais configuradas no container.

Para consultar a senha gerada automaticamente:

```bash
docker compose logs airflow | grep -i -A 5 "password"
```

Na interface do Airflow:

1. Localize a DAG `train_medical_abstracts_model`;
2. Ative a DAG, caso esteja pausada;
3. Consulte o próximo agendamento;
4. Acompanhe as tasks `ingest_data` e `train_model`.

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

## Executar a DAG manualmente

Para testar a DAG sem aguardar o agendamento mensal:

```bash
docker compose exec airflow \
  airflow dags test \
  train_medical_abstracts_model \
  2026-09-11
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

Os arquivos devem ter a data de modificação atualizada.

## Reiniciar o Airflow

Após alterar a DAG:

```bash
docker compose restart airflow
```

Se houver alteração no `Dockerfile` ou no `airflow/requirements.txt`:

```bash
docker compose down
docker compose build airflow
docker compose up -d airflow
```

## Volumes utilizados

O serviço do Airflow monta os diretórios do projeto no container:

```yaml
volumes:
  - ./airflow/dags:/opt/airflow/dags
  - ./airflow/logs:/opt/airflow/logs
  - ./airflow/plugins:/opt/airflow/plugins
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
