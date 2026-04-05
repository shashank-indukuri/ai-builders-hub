# Maintaaining Snowflake, MinIO, dbt connections

from dagster import ConfigurableResource, EnvVar
from core.minio_client import MinIOClient
from core.snowflake_client import SnowflakeClient

class MinIOResource(ConfigurableResource):
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str

    def get_client(self) -> MinIOClient:
        return MinIOClient(self.endpoint, self.access_key, self.secret_key, self.bucket)
    

class SnowflakeResource(ConfigurableResource):
    account: str
    user: str
    password: str
    database: str
    warehouse: str

    def get_client(self) -> SnowflakeClient:
        return SnowflakeClient(
            self.account, self.user, self.password, self.database, self.warehouse
        )

