# wires everything together
# The entry point script for DAGSTER

from dagster import Definitions, EnvVar
from dagster_dbt import DbtCliResource
from dagster_pipelines.assets.ingestion import bronze_sensor_telemetry
from dagster_pipelines.assets.dbt_assets import picarx_dbt_assets, DBT_PROJECT_DIR
from dagster_pipelines.assets.ml_training import surface_classifier_model
from dagster_pipelines.resources import MinIOResource, SnowflakeResource
from dagster_pipelines.schedules import hourly_pipeline
from dagster_pipelines.sensors import new_parquet_sensor


# we will import assets and resources here to get loaded into Dagster
defs = Definitions(assets=[bronze_sensor_telemetry, picarx_dbt_assets, surface_classifier_model],
                   resources={
                       "dbt": DbtCliResource(project_dir=str(DBT_PROJECT_DIR)),
                       "minio": MinIOResource(
                           endpoint="http://192.168.0.113:9002",
                           access_key="minioadmin",
                           secret_key="minioadmin",
                           bucket="picarx-bronze",
                       ),
                       "snowflake": SnowflakeResource(
                           account="<<YOUR_SNOWFLAKE_ACCOUNT_NAME>>",
                           user="<<YOUR_SNOWFLAKE_USER_ID",
                           password=EnvVar("SNOWFLAKE_PASSWORD"),
                           database="PICARX_DB",
                           warehouse="PICARX_WH",
                       ),
                   },
                   schedules=[hourly_pipeline],
                   sensors=[new_parquet_sensor],
                   )