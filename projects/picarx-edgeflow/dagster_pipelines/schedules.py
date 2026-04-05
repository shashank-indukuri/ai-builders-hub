# for scheduling purposes
from dagster import ScheduleDefinition, AssetSelection

hourly_pipeline = ScheduleDefinition(
    name="hourly_sensor_pipeline",
    target=AssetSelection.all(),
    cron_schedule="0 * * * *",   #every hour at :00
)