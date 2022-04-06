from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment
from azure.ml._constants import AssetTypes
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def eval_func(
    scored_data: ArtifactInput(type=AssetTypes.URI_FOLDER),
    label_col: str,
    eval_result: ArtifactOutput(type=AssetTypes.URI_FOLDER)
):
    # Load scored data
    result_df = pd.read_csv((Path(scored_data) / "scored_data.csv"))
    pred = result_df["predicted_val"]
    testy = result_df[label_col]

    # calculate the metrics of model
    metrics = {
        "Mean squared error" : mean_squared_error(testy, pred),
        "Coefficient of determination" : r2_score(testy, pred)
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metrics","value"])
    metrics_df.to_csv((Path(eval_result) / "eval_result.csv"), index=False)

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "py38.sklearn.conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

eval_component = dsl.command_component(
    name="eval_component",
    display_name="Eval",
    description="A sample command component to evaluate scored data.",
    version="0.0.1",
    # specify distribution type if needed
    # distribution={'type': 'mpi'},
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)(eval_func)