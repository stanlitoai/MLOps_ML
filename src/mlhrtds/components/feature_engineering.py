import os
from mlhrtds import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlhrtds.entity.config_entity import FeatureEngineeringConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_splitting(self):
        # Read the data from the specified path
        data = pd.read_csv(self.config.data_path)

        

        # Split the data into X (features) and y (target)
        X = data.drop([self.config.target_column], axis=1)
        y = data[[self.config.target_column]]
        
        # Define categorical and numerical columns
        categorical_columns = X.select_dtypes(include="object").columns
        numerical_columns = X.select_dtypes(exclude="object").columns

        # Perform one-hot encoding for categorical columns
        X = pd.get_dummies(X, columns=categorical_columns)
        print(X)
        # Define categorical and numerical columns
        categorical_columns = X.select_dtypes(include="object").columns
        numerical_columns = X.select_dtypes(exclude="object").columns

        # Split the data into training and test sets (75% train, 25% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        # Define transformers for numerical features (scaling in this case)
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Define transformers for categorical features (one-hot encoding)
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())  # You can customize options for encoding here
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])

        # Create the final data preprocessing pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        # Fit and transform the training data
        X_train_transformed = pipeline.fit_transform(X_train)
        print("X_train_transformed")
        print(X_train_transformed)
        # Transform the test data
        X_test_transformed = pipeline.transform(X_test)

        # Save the transformed data
        pd.DataFrame(X_train_transformed).to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        pd.DataFrame(X_test_transformed).to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)
        print("Splitted data into training and test sets and performed feature engineering.")

        