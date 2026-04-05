{{ config(materialized='table') }}


SELECT
    device_id,
    DATE_TRUNC('hour', event_timestamp)         AS hour,
    COUNT(*)                                    AS reading_count,
    AVG(distance_cm)                            AS avg_distance_cm,
    MIN(distance_cm)                            AS min_distance_cm,
    AVG(cpu_temp_celsius)                       AS avg_cpu_temp,
    MAX(cpu_temp_celsius)                       AS max_cpu_temp,
    SUM(CASE WHEN proximity_status = 'obstacle_near' THEN 1 ELSE 0 END) AS near_obstacle_count
FROM {{ ref('fct_sensor_readings') }}
GROUP BY device_id, DATE_TRUNC('hour', event_timestamp)