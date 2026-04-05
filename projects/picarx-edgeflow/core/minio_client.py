import io
import boto3
import pyarrow.parquet as pq
import pandas as pd

# Helper functions for accessing from Dagster Assests
class MinIOClient:
    def __init__(self, endpoint, access_key, secret_key, bucket):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
    # Listing ingested parquet files
    def list_parquet_files(self, prefix="sensors/telemetry/"):
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [
            obj["Key"] for obj in response.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
        
    def read_parquet_files(self, file_keys):
        dfs = []
        for key in file_keys:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            table = pq.read_table(io.BytesIO(obj["Body"].read()))
            dfs.append(table.to_pandas())
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)