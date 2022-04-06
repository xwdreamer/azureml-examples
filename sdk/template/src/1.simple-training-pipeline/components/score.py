from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment
from azure.ml._constants import AssetTypes
import pandas as pd
import pickle

def score_func(
    test_data: ArtifactInput(type=AssetTypes.URI_FOLDER),
    model_input: ArtifactInput(type=AssetTypes.CUSTOM_MODEL),
    feature_list: str,
    scored_data: ArtifactOutput(type=AssetTypes.URI_FOLDER)
):
    # Load testing data and init feature list
    test_df = pd.read_csv((Path(test_data) / "test_data.csv"))
    feature_col = feature_list.split(",")
    testX = test_df[feature_col]

    # Load model from pickle file
    model = pickle.load(open((Path(model_input) / "model.sav"), "rb"))

    # Predict data
    predictions = model.predict(testX)

    # Save to output folder
    test_df["predicted_val"] = predictions
    test_df.to_csv((Path(scored_data) / "scored_data.csv"), index=False)


# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "py38.sklearn.conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

score_component = dsl.command_component(
    name="score_component",
    display_name="Score",
    description="A sample command component to score data with input model.",
    version="0.0.1",
    # specify distribution type if needed
    # distribution={'type': 'mpi'},
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)(score_func)