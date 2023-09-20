from mlhrtds import logger
from mlhrtds.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlhrtds.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlhrtds.pipeline.stage_03_feature_engineering import FeatureEngineeringTrainingPipeline
from mlhrtds.pipeline.stage_04_model_training import *
# from mlds.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline




STAGE_NAME = "Data Ingestion stage"



try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Validation stage"

        
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Feature Enineering stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj = FeatureEngineeringTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
except Exception as e:
    logger.exception(e)
    raise e
    
    
    
    
STAGE_NAME = "Model Training stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
except Exception as e:
    logger.exception(e)
    raise e