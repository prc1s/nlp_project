from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_2_data_transformation import DataTransformationPipeline
from src.textSummarizer.pipeline.stage_3_model_trainer import ModelTrainerPipeline

try:
    stage_name = "Data Ingestion Stage"
    logger.info(f"Stage {stage_name} Initiated")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.initiate_data_ingestion()
    logger.info(f"{stage_name} Completed!")

    stage_name = "Data Transformation Stage"
    logger.info(f"Stage {stage_name} Initiated")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"{stage_name} Completed!")

    stage_name = "Model Training Stage"
    logger.info(f"Stage {stage_name} Initiated")
    model_training_pipeline = ModelTrainerPipeline()
    model_training_pipeline.initiate_model_training()
    logger.info(f"{stage_name} Completed!")

except Exception as e:
    logger.exception(e)
    raise e