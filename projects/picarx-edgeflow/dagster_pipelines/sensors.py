from dagster import sensor, RunRequest, SensorEvaluationContext
from dagster_pipelines.resources import MinIOResource

@sensor(
    target="bronze_sensor_telemetry",
    minimum_interval_seconds=60
)
def new_parquet_sensor(context: SensorEvaluationContext, minio: MinIOResource):
    mio = minio.get_client()
    files = mio.list_parquet_files()
    file_count = len(files)

    # cursor stores the last seen file count between sensor ticks
    last_count = int(context.cursor) if context.cursor else 0
    
    # we can modify this setting to have different condition also
    if file_count > last_count:
        context.log.info(f"New files detected: {file_count} (was {last_count})")
        context.update_cursor(str(file_count))
        yield RunRequest(run_key=f"parquet-{file_count}")
    else:
        context.log.info(f" No New files. still {file_count}.")