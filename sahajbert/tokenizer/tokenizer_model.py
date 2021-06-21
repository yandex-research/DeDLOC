from typing import Iterator, List, Union

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.models import Unigram


class SentencePieceUnigramCustomizedTokenizer(BaseTokenizer):
    """Custom SentencePiece Unigram Tokenizer with lower, digits, punctuation and Bengali characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(self, replacement: str = "▁", add_prefix_space: bool = True):
        tokenizer = Tokenizer(Unigram())

        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                normalizers.Replace("\u09e4", "\u0964"),
                normalizers.Replace("\u09e5", "\u0965"),
                normalizers.Replace("\u007c", "\u0964"),
                normalizers.Replace("\u09f7", "\u0964"),
                normalizers.Replace(Regex(r"(?<=[\u0980-\u09ff]):"), "\u0983"),
                normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
    ):
        """Train the model using the given files"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
    ):
        """Train the model using the given iterator"""

        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
