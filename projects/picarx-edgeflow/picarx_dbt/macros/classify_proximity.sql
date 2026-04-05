{% macro classify_proximity(column_name) %}
    CASE
        WHEN {{ column_name }} < 20 THEN 'obstacle_near'
        WHEN {{ column_name }} < 50 THEN 'obstacle_moderate'
        ELSE 'clear'
    END
{% endmacro %}