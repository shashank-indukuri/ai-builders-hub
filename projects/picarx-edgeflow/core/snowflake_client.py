import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Snowflake Helper functions for working with Snowflake from Dagster Assets
class SnowflakeClient:
    def __init__(self, account, user, password, database, warehouse):
        self.account = account
        self.user = user
        self.password = password
        self.database = database
        self.warehouse = warehouse
    
    def get_connection(self, schema):
        return snowflake.connector.connect(
            account = self.account,
            user = self.user,
            password = self.password,
            database = self.database,
            schema = schema,
            warehouse = self.warehouse
        )
    
    def get_max_offset(self, schema, table):
        conn = self.get_connection(schema)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COALESCE(MAX(KAFKA_OFFSET), -1) FROM {table}")
        max_offset = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return max_offset
    
    def load_dataframe(self, df, schema, table):
        conn = self.get_connection(schema)
        write_pandas(
            conn=conn, df=df, table_name=table,
            database = self.database, schema=schema,
        )
        conn.close()
        return len(df)