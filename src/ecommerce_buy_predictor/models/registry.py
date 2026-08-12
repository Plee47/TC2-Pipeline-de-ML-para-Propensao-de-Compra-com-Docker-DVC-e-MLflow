import mlflow


def promote_best_model_to_registry(
    model_name: str, metric_name: str = "f1"
) -> str:
    """Promote best run by metric to Model Registry.

    Args:
        model_name: Name to register the model as.
        metric_name: Metric to compare runs by.

    Returns:
        URI of promoted model version.
    """
    client = mlflow.MlflowClient()

    runs = mlflow.search_runs(order_by=[f"metrics.{metric_name} DESC"])
    if not runs.empty:
        best_run = runs.iloc[0]
        best_run_id = best_run.run_id

        model_uri = f"runs:/{best_run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        return f"models:/{model_name}/Production"

    raise ValueError("No runs found to promote")
