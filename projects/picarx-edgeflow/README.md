# PiCar-X EdgeFlow

A production-grade IoT data pipeline and ML inference system built on a SunFounder PiCar-X robot car. Sensor data flows from a Raspberry Pi through a full data stack, trains a surface classification model, and deploys it back to the edge for real-time inference.

## Architecture

```
Raspberry Pi 4 (PiCar-X)              Laptop (Docker + Local Services)
========================              ================================

pi_producer.py                        Kafka (KRaft, no Zookeeper)
  reads sensors every 0.5s               |
  publishes to Kafka ──────────────────> |
                                         |
                                         ├──> consumer_to_minio.py
                                                writes Parquet to MinIO
                                                (Hive-style date/hour partitioning)

                                      Dagster (orchestration)
                                      ───────────────────────
                                      bronze_sensor_telemetry (MinIO → Snowflake)
                                          ↓
                                      dbt build
                                        stg_sensors (view, SILVER)
                                          ↓
                                        fct_sensor_readings (incremental, GOLD)
                                          ↓
                                        agg_hourly_sensors (table, GOLD)
                                          ↓
                                        ml_surface_features (table, GOLD)
                                          ↓
                                      surface_classifier_model (Random Forest → MLflow)
                                          ↓
                                      ONNX export → deploy to Pi

pi_inference.py                       MLflow (experiment tracking)
  loads ONNX model                    Dagster UI (pipeline monitoring)
  real-time surface classification    dbt docs (data lineage)
  keyboard-controlled driving
```

## What This Project Demonstrates

| Concept | Implementation |
|---------|---------------|
| **Streaming ingestion** | Kafka producer on Pi, consumer groups for parallel processing |
| **Data lake** | MinIO (S3-compatible) with Parquet, Hive-style partitioning |
| **Medallion architecture** | Bronze (raw) → Silver (cleaned) → Gold (business logic) |
| **Data warehouse** | Snowflake with schema separation (BRONZE, SILVER, GOLD) |
| **Transformations** | dbt with incremental models, custom macros, source tests |
| **Orchestration** | Dagster with assets, resources, schedules, sensors |
| **Schema evolution** | Added 9 columns to running pipeline without downtime |
| **Feature engineering** | dbt window functions: rolling stats, cross-sensor ratios |
| **ML training** | scikit-learn Random Forest, 97% accuracy on 4 surface types |
| **Experiment tracking** | MLflow with parameter/metric logging, model registry |
| **Edge deployment** | ONNX export (237KB), onnxruntime inference at 2Hz on Pi |
| **Idempotency** | Kafka offset watermark prevents duplicate loading |
| **Retraining gate** | Dagster skips training if <500 new labeled rows |

## Prerequisites

### Hardware
- Raspberry Pi 4 (2GB+ RAM) with SunFounder PiCar-X kit
- Laptop/desktop for running the data stack
- Both on the same network

### Software — Laptop
- Docker & Docker Compose
- Python 3.10+
- Snowflake account (free trial works)

### Software — Raspberry Pi
- Raspberry Pi OS
- Python 3.x with `picarx` library (comes with SunFounder setup)

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Mohankandregula/ai-builders-hub.git
cd ai-builders-hub/projects/picarx-edgeflow

python -m venv .venv
source .venv/bin/activate

pip install kafka-python-ng boto3 pyarrow pandas
pip install snowflake-connector-python
pip install dbt-core dbt-snowflake dbt-utils
pip install dagster dagster-webserver dagster-snowflake dagster-dbt
pip install scikit-learn mlflow skl2onnx onnxruntime
```

### 2. Start infrastructure

```bash
docker compose up -d   # starts Kafka + MinIO
```

### 3. Create Kafka topics

```bash
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --create --topic picarx.sensors.telemetry \
  --partitions 3 --bootstrap-server localhost:9092
```

### 4. Configure Snowflake

Create the following in your Snowflake account:

```sql
CREATE WAREHOUSE PICARX_WH WITH WAREHOUSE_SIZE = 'XSMALL';
CREATE DATABASE PICARX_DB;
CREATE SCHEMA PICARX_DB.BRONZE;
CREATE SCHEMA PICARX_DB.SILVER;
CREATE SCHEMA PICARX_DB.GOLD;

CREATE TABLE PICARX_DB.BRONZE.RAW_SENSOR_TELEMETRY (
    DEVICE_ID VARCHAR,
    EVENT_TIMESTAMP VARCHAR,
    DISTANCE_CM FLOAT,
    GRAYSCALE_LEFT FLOAT,
    GRAYSCALE_CENTER FLOAT,
    GRAYSCALE_RIGHT FLOAT,
    CPU_TEMP FLOAT,
    KAFKA_PARTITION INT,
    KAFKA_OFFSET INT,
    KAFKA_TIMESTAMP TIMESTAMP,
    STEERING_ANGLE FLOAT,
    THROTTLE FLOAT,
    PAN_ANGLE FLOAT,
    TILT_ANGLE FLOAT,
    MODE VARCHAR,
    SURFACE_LABEL VARCHAR,
    SESSION_ID VARCHAR,
    WAYPOINT_ID INT,
    FRAME_PATH VARCHAR
);
```

### 5. Configure dbt

Create `~/.dbt/profiles.yml`:

```yaml
picarx_dbt:
  outputs:
    dev:
      account: YOUR_SNOWFLAKE_ACCOUNT
      database: PICARX_DB
      password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
      role: ACCOUNTADMIN
      schema: SILVER
      threads: 2
      type: snowflake
      user: YOUR_SNOWFLAKE_USER
      warehouse: PICARX_WH
  target: dev
```

### 6. Configure Dagster resources

Update `dagster_pipelines/definitions.py` with your Snowflake account and MinIO endpoint.

## Running the Pipeline

### Collect sensor data

```bash
# On the Pi:
source my_venv/bin/activate
python3 pi_producer.py                  # telemetry mode
python3 pi_producer.py carpet           # labeled surface collection

# On the laptop:
python3 consumer_to_minio.py            # Kafka → MinIO parquet
```

### Run the batch pipeline

```bash
# Start Dagster
export SNOWFLAKE_PASSWORD="your_password"
dagster dev -m dagster_pipelines.definitions

# Open http://localhost:3000
# Click "Materialize all" to run: ingestion → dbt → ML training
```

### Run dbt independently

```bash
cd picarx_dbt
dbt run          # build models
dbt test         # run data quality tests
dbt docs serve   # view lineage DAG
```

### Train the model manually

```bash
export SNOWFLAKE_PASSWORD="your_password"
python3 -m ML.train_surface_classifier
```

### Export and deploy model to Pi

```bash
# Export to ONNX
python3 -m ML.export_model

# Copy to Pi
scp ML/surface_classifier.onnx pi@<PI_IP>:~/Downloads/SF_FILES/
scp pi_inference.py pi@<PI_IP>:~/Downloads/SF_FILES/

# On the Pi:
pip install onnxruntime
python3 pi_inference.py    # real-time inference with keyboard driving
```

## Project Structure

```
picarx-edgeflow/
  docker-compose.yml              # Kafka + MinIO
  pi_producer.py                  # Sensor producer (runs on Pi)
  pi_inference.py                 # Live ML inference (runs on Pi)
  consumer_to_minio.py            # Kafka → MinIO parquet consumer
  
  core/
    minio_client.py               # Shared MinIO connection logic
    snowflake_client.py           # Shared Snowflake connection logic
  
  dagster_pipelines/
    definitions.py                # Dagster entry point
    resources.py                  # MinIO + Snowflake resource configs
    schedules.py                  # Hourly pipeline schedule
    sensors.py                    # MinIO new-file sensor
    assets/
      ingestion.py                # MinIO → Snowflake bronze asset
      dbt_assets.py               # dbt project as Dagster assets
      ml_training.py              # ML training asset with retraining gate
  
  picarx_dbt/
    models/
      staging/stg_sensors.sql     # Bronze → Silver (rename, cast, clean)
      marts/fct_sensor_readings.sql   # Incremental fact table
      marts/agg_hourly_sensors.sql    # Hourly aggregations
      ml/ml_surface_features.sql      # ML feature engineering
    macros/
      classify_proximity.sql      # Reusable distance classification
      generate_schema_name.sql    # Custom schema routing
  
  ML/
    train_surface_classifier.py   # Training script with MLflow logging
    export_model.py               # MLflow → ONNX export
```

## Model Performance

Surface classifier trained on ~1,500 labeled sensor readings across 4 surfaces:

| Surface | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Cardboard | 1.00 | 0.99 | 0.99 |
| Carpet | 0.96 | 0.99 | 0.97 |
| Ceramic | 0.94 | 0.95 | 0.94 |
| Mat | 0.99 | 0.96 | 0.97 |
| **Overall** | **0.97** | **0.97** | **0.97** |

Top features by importance: rolling grayscale means (68.5%), raw grayscale mean (14.7%), rolling standard deviations (9.4%).

## Roadmap

- [ ] LLM tool calling for autonomous reasoning (Phase 3)
- [ ] Grafana dashboard for pipeline monitoring (Phase 4)
- [ ] GitHub Actions CI/CD (Phase 5)
- [ ] Behavioral cloning for autonomous driving
- [ ] Schema Registry (Avro) for Kafka data contracts