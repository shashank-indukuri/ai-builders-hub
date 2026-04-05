{{ config(
    materialized='incremental',
    unique_key='kafka_offset'
) }}

SELECT
    device_id,
    event_timestamp,
    distance_cm,
   {{ classify_proximity('distance_cm') }} AS proximity_status,
    grayscale_left,
    grayscale_center,
    grayscale_right,
    cpu_temp_celsius,
    CASE
        WHEN cpu_temp_celsius > 70 THEN TRUE
        ELSE FALSE
    END AS is_overheating,

    -- control state
    steering_angle,
    throttle,
    pan_angle,
    tilt_angle,

    -- session metadata
    mode,
    surface_label,
    session_id,
    waypoint_id,
    frame_path,
    
    kafka_offset
FROM {{ ref('stg_sensors') }}

{% if is_incremental() %}
    WHERE kafka_offset > (SELECT MAX(kafka_offset) FROM {{ this }})
{% endif %}