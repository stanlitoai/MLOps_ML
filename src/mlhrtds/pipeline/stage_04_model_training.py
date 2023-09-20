from mlhrtds.config.configuration import ConfigurationManager
from mlhrtds.components.model_training import *
from mlhrtds import logger




STAGE_NAME = "Model Training stage"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
        logger.info(f"Model Training stage completed!")
            
        
        

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n\nx====================================================x")
    except Exception as e:
        logger.exception(e)
        raise e