from pathlib import Path
from azure.ml import dsl, ArtifactInput, ArtifactOutput
from azure.ml.entities import Environment
from azure.ml._constants import AssetTypes
import pandas as pd

def data_prep_func(
    data_source: ArtifactInput(type=AssetTypes.URI_FILE),
    data_cooked: ArtifactOutput(type=AssetTypes.URI_FOLDER),
):
    # Load input data into df
    input_df = pd.read_csv(data_source)

    # Data processing
    ## replace here...
    input_df = input_df.astype(
        {
            "pickup_longitude": "float64",
            "pickup_latitude": "float64",
            "dropoff_longitude": "float64",
            "dropoff_latitude": "float64",
        }
    )

    ## Filter data
    input_df = input_df[
        (input_df.pickup_longitude <= -73.72)
        & (input_df.pickup_longitude >= -74.09)
        & (input_df.pickup_latitude <= 40.88)
        & (input_df.pickup_latitude >= 40.53)
        & (input_df.dropoff_longitude <= -73.72)
        & (input_df.dropoff_longitude >= -74.72)
        & (input_df.dropoff_latitude <= 40.88)
        & (input_df.dropoff_latitude >= 40.53)
    ]

    input_df = input_df.dropna(how="all")

    # Save intermediate data
    input_df.to_csv((Path(data_cooked) / "cooked_data.csv"), index=False)

# init customer environment with conda YAML
# the YAML file shall be put under your code folder.
conda_env = Environment(
    conda_file=Path(__file__).parent / "py38.sklearn.conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

data_prep_component = dsl.command_component(
    name="data_prep_component",
    display_name="Data Preparation",
    description="A sample command component to do data preparation.",
    version="0.0.1",
    # specify customer environment, note that azure-ml must be included.
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'
)(data_prep_func)