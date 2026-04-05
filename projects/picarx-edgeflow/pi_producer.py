#PiCar-X Kafka Producer
# This script runs ON the Raspberry Pi.
# It reads sensor data from PiCar-X and sends it to Kafka running on docker

import json
import time
from datetime import datetime, timezone
import sys

from kafka import KafkaProducer
from picarx import Picarx


# CONFIGURATION

KAFKA_BROKER = "192.168.0.123:9092"   # laptop's IP where Kafka is running
TOPIC = "picarx.sensors.telemetry"    # Which mailbox to drop messages into
DEVICE_ID = "picarx-01"               # Unique name for this robot
READ_INTERVAL = 0.5                   # Read sensors every 0.5 seconds (2 Hz)

# session configuration - set these before each collection run
MODE = "surface_collection"                    # "telemetry" for normal, "patrol" for patrol runs, "manual_drive" for behavioral cloning
SURFACE_LABEL = sys.argv[1] if len(sys.argv) > 1 else None                  # set to "carpet", "tile", "wood" etc during surface classification runs
SESSION_ID = f"surface_{SURFACE_LABEL}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


# PRODUCER SETUP
producer = KafkaProducer(

    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),    # Kafka only understands raw bytes, not Python objects.
    key_serializer=lambda k: k.encode("utf-8") if k else None,      # Keys determine WHICH PARTITION a message goes to.
    acks="all",   # ALL replicas confirm.
    linger_ms=100,
    batch_size=16384, # don't send messages one-by-one, group them instruction
    retries=3, # if sending fails, try again
)

# INITIALIZE PICAR-X
px = Picarx()
print(f"PiCar-X initialized. Sending sensor data to {KAFKA_BROKER}")
print(f"Topic: {TOPIC} | Device: {DEVICE_ID} | Interval: {READ_INTERVAL}s")
print("Press Ctrl+C to stop.\n")


try:
    message_count = 0

    # track the current control state what is set
    current_steering = 0
    current_throttle = 0
    current_pan = 0
    current_tilt = 0

    px.forward(20) #slow speed for collecting training data for surface classification
    current_throttle = 20

    while True:
        # Read all sensors
        distance = px.get_distance()
        grayscale = px.get_grayscale_data()

        # Read CPU temperature, Pi'S system health
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = round(int(f.read().strip()) / 1000, 1)
        except Exception:
            cpu_temp = -1

        # Build the message OR the event
        message = {
            "device_id": DEVICE_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ultrasonic": {
                "distance_cm": round(distance, 2)
            },
            "grayscale": {
                "left": grayscale[0],
                "center": grayscale[1],
                "right": grayscale[2]
            },
            "system": {
                "cpu_temp": cpu_temp
            },
            "control": {
                "steering_angle": current_steering,
                "throttle": current_throttle,
                "pan_angle": current_pan,
                "tilt_angle": current_tilt
            },
            "meta": {
                "mode": MODE,
                "surface_label": SURFACE_LABEL,
                "session_id": SESSION_ID,
                "waypoint_id": None,
            },
        }

        future = producer.send(
            topic=TOPIC,
            key=DEVICE_ID,
            value=message
        )

        message_count += 1

        # Print every 10th message to avoid printing everything on the terminal
        if message_count % 10 == 0:
            print(f"[{message_count}] distance={distance:.1f}cm | "
                  f"grayscale={grayscale} | cpu={cpu_temp}°C")

        time.sleep(READ_INTERVAL)

except KeyboardInterrupt:
    print(f"\nStopping. Sent {message_count} messages total.")

finally:
    producer.flush()  # ALL buffered messages out to Kafka before we exit
    producer.close()
    px.stop()
    print("Producer closed cleanly.")
