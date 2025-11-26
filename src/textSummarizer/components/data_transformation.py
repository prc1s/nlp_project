import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.textSummarizer.entity import DataTransformationConfig

class DataTransformation():
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        logger.info("Entered DataTransformation Class convert_examples_to_features Method")
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

            logger.info(f"Converted Batch:\n\tinput_ids: {input_encodings['input_ids']}\n\tattention_mask: {input_encodings['attention_mask']}\n\tlabels: {target_encodings['input_ids']}")

            return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
        

    def convert(self):
        logger.info("Entered DataTransformation Class convert Method")
        dataset = load_from_disk(self.config.data_path)
        converted_dataset = dataset.map(self.convert_examples_to_features, batched=True)
        converted_dataset.save_to_disk(os.path.join(self.config.root_dir, "converted_samsum_dataset"))
        logger.info(f"Converted dataset at", os.path.join(self.config.root_dir, "converted_samsum_dataset"))
