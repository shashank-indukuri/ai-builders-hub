# Python Script to upload data into MinIO.

import json
import time
from datetime import datetime, timezone
from io import BytesIO

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from kafka import KafkaConsumer



# --- kafka Configuration ---
KAFKA_BROKER = "localhost:9092"
TOPIC = "picarx.sensors.telemetry"
GROUP_ID = "picarx-minio-sink"

# --- MinIO Configuration ---
MINIO_ENDPOINT = "http://192.168.0.123:9002"    # currently in wsl, minio docker is running on laptop, hence using laptop's IP address.
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "picarx-bronze"           # Bronze = raw, unprocessed data (medallion architecture)

# --- Batching Configuration ---
BATCH_SIZE = 50                       # Collect 50 messages, then write 1 parquet file to MinIO

print(f"Starting Kafka consumer to read from topic '{TOPIC}' and write to MinIO bucket '{BUCKET_NAME}'...\n")

# --- MinIO Setup ---
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# create bucket if it doesn't exist
try:
    s3.head_bucket(Bucket=BUCKET_NAME)
    print(f"Bucket '{BUCKET_NAME}' already exists.")
except Exception:
    s3.create_bucket(Bucket=BUCKET_NAME)
    print(f"Bucket '{BUCKET_NAME}' created.")

# --- Kafka Consumer Setup ---
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    group_id=GROUP_ID,
    auto_offset_reset="earliest",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
)

print(f"Consumer '{GROUP_ID}' started. Reading messages from topic '{TOPIC}'...\n")
print(f"Will write a Parquet file to MinIO every {BATCH_SIZE} messages.\n")

# --- Main Loop ---
batch = []
file_count = 0

try:
    print("Waiting for messages from Kafka... and for Partition assignment ( can take 10-30 secinds on first run )...")
    batch = []
    for message in consumer:
        if len(batch) == 0 and file_count == 0:
            print("Partition assigned. Pipeline is flowing!. Starting to consume messages...\n")
        data = message.value

        # Flatten the nested JSON into a flat row
        # (Parquet works best with flat structures, columnar data)
        row = {
            "device_id": data.get("device_id"),
            "timestamp": data.get("timestamp"),
            "distance_cm": data["ultrasonic"]["distance_cm"],
            "grayscale_left": data["grayscale"]["left"],
            "grayscale_center": data["grayscale"]["center"],
            "grayscale_right": data["grayscale"]["right"],
            "cpu_temp": data["system"]["cpu_temp"],
            "kafka_partition": message.partition,
            "kafka_offset": message.offset,

            # control state
            "steering_angle": data.get("control", {}).get("steering_angle"),
            "throttle": data.get("control", {}).get("throttle"),
            "pan_angle": data.get("control", {}).get("pan_angle"),
            "tilt_angle": data.get("control", {}).get("tilt_angle"),

            # session metadata
            "mode": data.get("meta", {}).get("mode"),
            "surface_label": data.get("meta", {}).get("surface_label"),
            "session_id": data.get("meta", {}).get("session_id"),
            "waypoint_id": data.get("meta", {}).get("waypoint_id"),
            "frame_path": data.get("camera", {}).get("frame_path")
        }
        batch.append(row)

        # When batch is full, write a Parquet file to MinIO
        if len(batch) >= BATCH_SIZE:
            # Convert list of dicts -> Pyarrow Table -> Parquet bytes
            table = pa.Table.from_pylist(batch)
            buffer = BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)

            # build the s3 key (file path) with date partitioning
            now = datetime.now(timezone.utc)
            key = (
                f"sensors/telemetry/"
                f"dt={now.strftime('%Y-%m-%d')}/"
                f"hr={now.strftime('%H')}/"
                f"batch_{file_count:04d}.parquet"
            )

            # Upload to MinIO
            s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue())

            file_count += 1
            print(f"[File {file_count}] Wrote {len(batch)} rows -> s3://{BUCKET_NAME}/{key}")

            batch = []  # Clear batch for next set of messages
except KeyboardInterrupt:
    # Write any remaining messages in batch before exiting
    if batch:
        table = pa.Table.from_pylist(batch)
        buffer = BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        now = datetime.now(timezone.utc)
        key = (
            f"sensors/telemetry/"
            f"dt={now.strftime('%Y-%m-%d')}/"
            f"hr={now.strftime('%H')}/"
            f"batch_{file_count:04d}_final.parquet"
        )

        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue())
        print(f"\n[Final File] Wrote remaining {len(batch)} rows -> s3://{BUCKET_NAME}/{key}")
    
    print(f"\n Done. Wrote {file_count+1} parquet files total. Consumer stopped.")
finally:
    consumer.close()