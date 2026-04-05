{{ config(materialized='view') }}

SELECT
    DEVICE_ID                                   AS device_id,
    TO_TIMESTAMP(EVENT_TIMESTAMP)               AS event_timestamp,
    DISTANCE_CM::FLOAT                          AS distance_cm,
    GRAYSCALE_LEFT::INT                         AS grayscale_left,
    GRAYSCALE_CENTER::INT                       AS grayscale_center,
    GRAYSCALE_RIGHT::INT                        AS grayscale_right,
    CPU_TEMP::FLOAT                             AS cpu_temp_celsius,
    KAFKA_PARTITION::INT                        AS kafka_partition,
    KAFKA_OFFSET::BIGINT                        AS kafka_offset,

    -- control state (NULL for pre schema-evolution rows)
    STEERING_ANGLE::FLOAT                       AS steering_angle,
    THROTTLE::FLOAT                             AS throttle,
    PAN_ANGLE::FLOAT                            AS pan_angle,
    TILT_ANGLE::FLOAT                           AS tilt_angle,

    -- session metadata
    MODE                                        AS mode,
    SURFACE_LABEL                               AS surface_label,
    SESSION_ID                                  AS session_id,
    WAYPOINT_ID::INT                            AS waypoint_id,
    FRAME_PATH                                  AS frame_path

FROM {{ source('bronze', 'RAW_SENSOR_TELEMETRY') }}