import mlflow
import mlflow.tracking as tracking

# Set the MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:8000")  # Replace with your MLflow Tracking server URL

# Start an MLflow run
mlflow.start_run()

# Log metrics and artifacts
mlflow.log_metric("accuracy", 0.3)
# mlflow.log_artifact("mode1l_checkpoint.pth")

# Get the URL for the MLflow Tracking UI
tracking_uri = mlflow.get_tracking_uri()
print(f"MLflow Tracking UI: {tracking_uri}")

# End the MLflow run
mlflow.end_run()