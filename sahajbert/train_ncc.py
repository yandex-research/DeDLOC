# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Source: https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from tokenization_albert_bengali_fast import AlbertBengaliTokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(default="ncc", metadata={"help": "The name of the task to train on: ncc"})
    dataset_name: Optional[str] = field(
        default="indic_glue", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="sna.bn", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


@dataclass
class AdditionalTrainingArguments:
    early_stopping_patience: int = field(
        default=1,
        metadata={"help": "The number of evaluation calls to wait before stopping training while metric worsens."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "How much the metric must improve to satisfy early stopping conditions."},
    )


def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalTrainingArguments))
    model_args, data_args, training_args, additional_training_args = parser.parse_args_into_dataclasses()
    training_args.do_train = True
    training_args.do_eval = True
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.evaluation_strategy = "epoch"
    return model_args, data_args, training_args, additional_training_args


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")


def run(model_args, data_args, training_args, additional_training_args):
    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets.
    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)

    # Labels
    text_column_name = "text"
    label_column_name = "label"
    label_list = datasets["train"].features[label_column_name].names
    num_labels = len(label_list)
    # No need to convert the labels since they are already ints.
    label_to_id = {i: i for i in range(num_labels)}

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )
    if model_args.model_name_or_path == "neuropark/sahajBERT":
        tokenizer = AlbertBengaliTokenizerFast.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Preprocessing the datasets
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples[text_column_name], padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(preprocess_function, batched=True)

    valid_dataset = datasets["validation"]
    valid_dataset = valid_dataset.map(preprocess_function, batched=True)

    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding
    data_collator = default_data_collator if data_args.pad_to_max_length else None

    # Metrics
    metric = load_metric("accuracy")

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=additional_training_args.early_stopping_patience,
        early_stopping_threshold=additional_training_args.early_stopping_threshold,
    )
    callbacks = [early_stopping]

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Training
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    metrics["eval_samples"] = len(test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    model_args, data_args, training_args, additional_training_args = parse_arguments()

    run(model_args, data_args, training_args, additional_training_args)


if __name__ == "__main__":
    main()
