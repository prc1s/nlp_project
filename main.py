from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline

stage_name = "Data Ingestion Stage"

try:
    logger.info(f"Stage {stage_name} Initiated")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.initiate_data_ingestion()
    logger.info(f"{stage_name} Completed!")
except Exception as e:
    logger.exception(e)
    raise e