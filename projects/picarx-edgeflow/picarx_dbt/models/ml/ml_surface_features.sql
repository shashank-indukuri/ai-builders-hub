{{ config(materialized='table',
          schema='GOLD') }}

-- carpet and ceramic overlap in raw grayscale values (both 800-1400) noticed from the car driving results. 
-- We need features that separate them - variance (carpet is textured, ceramic is smooth) and 
-- sensor spread (how different are grayscale left/center/right sensor readings from each other).

WITH windowed AS (
    SELECT
        device_id,
        event_timestamp,
        kafka_offset,
        surface_label,
        session_id,
        distance_cm,
        grayscale_left,
        grayscale_center,
        grayscale_right,

        -- Feature: mean grayscale (to find out overall reflectance, it depends on color and surface texture also)
        (grayscale_left + grayscale_center + grayscale_left) / 3.0 AS gs_mean,

        -- Feature: spread across sensors (to find out uneven surface detection)
        GREATEST(grayscale_left, grayscale_center, grayscale_right)
        - LEAST(grayscale_left, grayscale_center, grayscale_right)
            AS gs_spread, -- if low then a rigid floor, if high ^ kind surface
        
        -- Feature: rolling standard deviation over 10 readings( to find out the texture roughness for 10 cm eg along its way from left sensor only)
        STDDEV(grayscale_left) OVER (
            PARTITION BY device_id, session_id ORDER BY event_timestamp  -- because, we want to extract all features per texture/session only session1 carpet, session2 ceramic etc
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_left_rolling_std_10,  -- if zigzag then either surface is rough like carpet spikes, for rigid floor will get nearly same readings


        -- Feature: rolling standard deviation over 10 readings( to find out the texture roughness for 10 cm eg along its way from center sensor only)
        STDDEV(grayscale_center) OVER (
            PARTITION BY device_id, session_id ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_center_rolling_std_10,  -- if zigzag then either surface is rough like carpet spikes, for rigid floor will get nearly same readings


        -- Feature: rolling standard deviation over 10 readings( to find out the texture roughness for 10 cm eg along its way from right sensor only)
        STDDEV(grayscale_right) OVER (
            PARTITION BY device_id, session_id ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_right_rolling_std_10,  -- if zigzag then either surface is rough like carpet spikes, for rigid floor will get nearly same readings

        
        -- Feature: rolling mean over 10 readings ( same as above but to find the avg smoothed signal)
        AVG(grayscale_left) OVER(
            PARTITION BY device_id, session_id ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_left_rolling_mean_10,

        -- Feature: rolling mean over 10 readings ( same as above but to find the avg smoothed signal)
        AVG(grayscale_center) OVER(
            PARTITION BY device_id, session_id ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_center_rolling_mean_10,

        -- Feature: rolling mean over 10 readings ( same as above but to find the avg smoothed signal)
        AVG(grayscale_right) OVER(
            PARTITION BY device_id, session_id ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS gs_right_rolling_mean_10,

        -- Feature: ratio of center to outer sensors (surface curvature/angle , __ or ^ or V shape surfaces etc)
        CASE
            WHEN (grayscale_left + grayscale_right) > 0
            THEN grayscale_center / ((grayscale_left + grayscale_right) / 2.0)
            ELSE NULL
        END AS gs_center_to_outer_ratio
    
    FROM {{ ref('fct_sensor_readings') }}
    WHERE surface_label IS NOT NULL -- as the data has some pre collected data with NULLs
)

SELECT
    device_id,
    event_timestamp,
    kafka_offset,
    surface_label,
    session_id,

    -- Raw sensor values
    grayscale_left,
    grayscale_center,
    grayscale_right,
    distance_cm,

    -- Computed features
    gs_mean,
    gs_spread,
    gs_left_rolling_std_10,
    gs_center_rolling_std_10,
    gs_right_rolling_std_10,
    gs_left_rolling_mean_10,
    gs_center_rolling_mean_10,
    gs_right_rolling_mean_10,
    gs_center_to_outer_ratio,

    -- combined texture feature (avg rolling std across all 3 sensors, to find out relation b/w each sensor)
    (gs_left_rolling_std_10 + gs_center_rolling_std_10 + gs_right_rolling_std_10) / 3.0
        AS gs_texture_score -- if low then rigid floor as deviation won't be there for 10 redings, else carpet kind of surface

FROM windowed
WHERE gs_left_rolling_std_10 is NOT NULL  -- dropping first 9 rows per session (incomplete window) because, we need atleast 10 rows to find the features stats