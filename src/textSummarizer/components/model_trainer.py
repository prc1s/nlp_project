from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import torch, sys
from datasets import load_from_disk
from src.textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.config = model_trainer_config

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        data_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps, save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradiant_accumulation_steps,

            no_cuda=True,
            use_mps_device=False,

        )

        trainer = Trainer(model=model_pegasus, args=trainer_args, tokenizer=tokenizer,
                          data_collator=seq2seq_data_collator, train_dataset=data_samsum_pt['test'],
                          eval_dataset=data_samsum_pt['validation'])
        
        trainer.train()

        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

        
