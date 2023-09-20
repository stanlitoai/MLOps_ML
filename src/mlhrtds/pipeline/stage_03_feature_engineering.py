from mlhrtds.config.configuration import ConfigurationManager
from mlhrtds.components.feature_engineering import DataTransformation
from mlhrtds import logger




STAGE_NAME = "Feature Enineering stage"


class FeatureEngineeringTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config =  ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transform = DataTransformation(config=data_transformation_config)
        data_transform.train_test_splitting()
        logger.info(f"Feature Enineering stage completed!")
            
        
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = FeatureEngineeringTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
    except Exception as e:
        logger.exception(e)
        raise e