import pandas as pd
import os
from mlhrtds import logger
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from mlhrtds.entity.config_entity import *


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data
        # test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = test_data
        print(train_y)
        # test_y = test_data[[self.config.target_column]]


        cr = CatBoostRegressor(verbose=False)
        parameters = {'depth' : [6,8,10],
              'learning_rate' : [0.01, 0.05, 0.1],
              'iterations'    : [30, 50, 100]
              }
        grid = GridSearchCV(estimator=cr, param_grid = parameters, cv = 2, n_jobs=-1)
        grid.fit(train_x, train_y)

        joblib.dump(grid, os.path.join(self.config.root_dir, self.config.model_name))
