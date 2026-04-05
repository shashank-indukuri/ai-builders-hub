# MinIO to Snowflake asset

import io
import boto3
import pyarrow.parquet as pq
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dagster import asset, EnvVar, MetadataValue, AssetExecutionContext
from dagster_pipelines.resources import MinIOResource, SnowflakeResource

@asset(
    description="Incrementally loads new sensor parquet data from MinIO into snowflake bronze layer"
)
def bronze_sensor_telemetry(context: AssetExecutionContext,
                            minio: MinIOResource,
                            snowflake: SnowflakeResource,
                            ):
    
    # MinIO: reading parquet files
    mio = minio.get_client()
    sf = snowflake.get_client()

    #Fetching the Watermark: What's the highest offset already in Snowflake? to avoid loading duplicates into snowflake
    max_offset = sf.get_max_offset("BRONZE", "RAW_SENSOR_TELEMETRY")
    context.log.info(f"Current max kafka_offset in Snowflake: {max_offset}")

    # Reading all parquet files from MinIO
    files = mio.list_parquet_files()
    df = mio.read_parquet_files(files)

    # Filtering the files which are newer than watermark 
    df = df[df["kafka_offset"] > max_offset]
    context.log.info(f"New rows after filtering: {len(df)}")

    if len(df) == 0:
        context.log.info("No new data. Skipping load.")
        context.add_output_metadata({
            "row_count": MetadataValue.int(0),
            "files_scanned": MetadataValue.int(len(files)),
            "status": MetadataValue.text("skipped - no nw data"),
        })
        return

    # Prepare columns to match to Snowflake schema
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "event_timestamp"})
    if "kafka_timestamp" not in df.columns:
        df["kafka_timestamp"] = df["event_timestamp"]
    df.columns = [c.upper() for c in df.columns]

    # Snowflake: bulk load only the new rows to Snowflake
    rows_loaded = sf.load_dataframe(df, "BRONZE", "RAW_SENSOR_TELEMETRY")

    context.add_output_metadata(
        {
            "row_count": MetadataValue.int(len(df)),
            "files_processed": MetadataValue.int(len(files)),
            "prev_max_offset": MetadataValue.int(int(max_offset)),
            "new_max_offset" : MetadataValue.int(int(df["KAFKA_OFFSET"].max())),
        }
    )