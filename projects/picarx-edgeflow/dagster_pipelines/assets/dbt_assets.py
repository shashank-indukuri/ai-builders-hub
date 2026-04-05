# Wrapping your dbt project into dagster

from pathlib import Path
from dagster import AssetExecutionContext
from dagster_dbt import DbtCliResource, dbt_assets, DbtProject


DBT_PROJECT_DIR = Path(__file__).resolve().parents[2] / "picarx_dbt"

dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR)

@dbt_assets(manifest=dbt_project.manifest_path)
def picarx_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()