import os, sys
import urllib.request as request
import zipfile
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig

class DataIngestion():
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def download_file(self):
        logger.info("Entered DataIngestion Class download_file Method")
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading file from {self.config.source_URL}")
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"File Saved at {self.config.local_data_file}")
        else:
            logger.info(f"{self.config.local_data_file} Already Exists!")

    def extract_zip(self):
        logger.info("Entered DataIngestion Class extract_zip Method")
        unzip_path = self.config.unzip_dir
        os.makedirs(self.config.unzip_dir, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info(f"file unziped at {unzip_path}")