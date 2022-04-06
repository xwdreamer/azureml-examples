from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment
from azure.ml._constants import AssetTypes
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data_func(
    data_cooked: ArtifactInput(type=AssetTypes.URI_FOLDER),
    test_size: float,
    random_state: int,
    data_train: ArtifactOutput(type=AssetTypes.URI_FOLDER),
    data_test: ArtifactOutput(type=AssetTypes.URI_FOLDER),
):
    # Load input data into df
    input_df = pd.read_csv((Path(data_cooked) / "cooked_data.csv"))

    # Split data by input parameters
    train_data, test_data = train_test_split(input_df, test_size=test_size, random_state=random_state)

    # Save intermediate data
    train_data.to_csv((Path(data_train) / "train_data.csv"), index=False)
    test_data.to_csv((Path(data_test) / "test_data.csv"), index=False)

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "py38.sklearn.conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

split_data_component = dsl.command_component(
    name="data_split_component",
    display_name="Split Data",
    description="A sample command component to split data.",
    version="0.0.1",
    # specify distribution type if needed
    # distribution={'type': 'mpi'},
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)(split_data_func)