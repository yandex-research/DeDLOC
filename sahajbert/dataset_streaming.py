"""
This module can create a "lazy" HF dataset for bengali.
Lazy dataset can yield samples before the downloads samples in background while it is iterating.
"""
import random
import logging
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from typing import Sequence, Optional

import torch
from bnlp import NLTKTokenizer
from datasets import load_dataset
from datasets.streaming import load_dataset, merge_datasets
from transformers import AlbertTokenizerFast, AlbertTokenizer

logger = logging.getLogger(__name__)
bnlp_separator = NLTKTokenizer()


def create_instances_from_document(tokenizer, document, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    segmented_sents = bnlp_separator.sentence_tokenize(document.replace("।", " । "))

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 1:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []

                for j in range(a_end, len(current_chunk)):
                    tokens_b.append(current_chunk[j])

                if random.random() < 0.5:
                    # Random next
                    is_random_next = True
                    # in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instance = tokenizer(
                    " ".join(tokens_a),
                    " ".join(tokens_b),
                    truncation="longest_first",
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert len(instance["input_ids"]) <= max_seq_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


def tokenize_function(tokenizer, examples):
    # Remove empty texts
    texts = (text for text in examples["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=512)
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples


class WrappedIterableDataset(torch.utils.data.IterableDataset):
    """Wraps huggingface IterableDataset as pytorch IterableDataset, implement default methods for DataLoader"""

    def __init__(self, hf_iterable, verbose: bool = True):
        self.hf_iterable = hf_iterable
        self.verbose = verbose

    def __iter__(self):
        started = False
        logger.info("Pre-fetching training samples...")
        while True:
            for sample in self.hf_iterable:
                if not started:
                    logger.info("Began iterating minibatches!")
                    started = True
                yield sample


def make_lazy_wikioscar_dataset(
    tokenizer,
    probs: Sequence[float] = (0.23, 0.77),
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 256,
):
    wiki = load_dataset("lhoestq/wikipedia_bn", split="train")
    # ^-- no need for script_version: already compatible with streaming

    oscar = load_dataset("oscar", "unshuffled_deduplicated_bn", split="train", script_version="streaming")

    # both should have the same columns
    wiki = wiki.map(lambda x: {"text": x["text"], "orig": f"wiki[{x['title']}]"})
    oscar = oscar.map(lambda x: {"text": x["text"], "orig": f"oscar[{x['id']}]"})

    # merge, shuffle and set pytorch format
    dataset = merge_datasets([wiki, oscar], probabilities=list(probs))
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    # ^-- this creates a buffer of random examples that will be refilled in background

    dataset = dataset.map(partial(tokenize_function, tokenizer), batch_size=preprocessing_batch_size)
    dataset = dataset.with_format("torch")
    return WrappedIterableDataset(dataset)
