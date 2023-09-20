from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mlhrtds.utils.common import read_yaml, create_directories

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    data_path: Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath="config.yaml",  # Update with your config file path
        params_filepath="params.yaml",  # Update with your params file path
        schema_filepath="schema.yaml",  # Update with your schema file path
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_transformation_config(self) -> FeatureEngineeringConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = FeatureEngineeringConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config

class DataTransformation:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Perform feature engineering here
        # For example, you can handle missing values, encode categorical features, and scale numerical features
        # Let's assume 'categorical_columns' and 'numerical_columns' contain the column names

        categorical_columns = []  # Update with your categorical column names
        numerical_columns = []    # Update with your numerical column names

        # Split the data into training and test sets. (0.75, 0.25) split.
        X = data.drop(columns=categorical_columns + numerical_columns)
        y = data[config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Create transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())  # You can add more transformers here
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # You can add more transformers here
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])

        # Create the final pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        # Save the transformed data
        pd.DataFrame(X_train_transformed).to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        pd.DataFrame(X_test_transformed).to_csv(os.path.join(self.config.root_dir, "valid.csv"), index=False)

        print("Splitted data into training and test sets and performed feature engineering.")

try:
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transform = DataTransformation(config=data_transformation_config)
    data_transform.train_test_splitting()
except Exception as e:
    raise e
