from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logger

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info(">>>>> Training pipeline started <<<<<")
            config_manager = ConfigurationManager()
            
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(data_validation_config)
            val_status = data_validation.validate_all_columns(train_data_path)

            if not val_status:
                logger.error("Data validation failed")
                raise Exception("Data validation failed. Schema mismatch.")

            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(data_transformation_config)
            data_transformation.initiate_data_transformation()

            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)
            model_trainer.initiate_model_trainer()

            model_eval_config = config_manager.get_model_evaluation_config()
            model_eval = ModelEvaluation(model_eval_config)
            model_eval.initiate_model_evaluation()
            
            logger.info(">>>>> Training pipeline completed successfully <<<<<")
        except Exception as e:
            logger.error(f"Error executing Training Pipeline: {str(e)}")
            raise e

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
