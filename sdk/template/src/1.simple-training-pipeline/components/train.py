from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment
from azure.ml._constants import AssetTypes
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pickle

def train_func(
    train_data: ArtifactInput(type=AssetTypes.URI_FOLDER),
    feature_list: str,
    label_col: str,
    learning_rate: float,
    n_estimators: int,
    model_output: ArtifactOutput(type=AssetTypes.CUSTOM_MODEL)
):
    # Load training data and init feature list
    train_df = pd.read_csv((Path(train_data) / "train_data.csv"))
    feature_col = feature_list.split(",")

    trainy = train_df[label_col]
    trainX = train_df[feature_col]

    # Train model by using linear regression from sklearn
    model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators).fit(trainX, trainy)

    # Save model to output
    pickle.dump(model, open((Path(model_output) / "model.sav"), "wb"))

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "py38.sklearn.conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

train_component = dsl.command_component(
    name="data_train_component",
    display_name="Train",
    description="A sample command component to train model.",
    version="0.0.1",
    # specify distribution type if needed
    # distribution={'type': 'mpi'},
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)(train_func)