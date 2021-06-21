from typing import Optional, Tuple

from transformers import AddedToken, PreTrainedTokenizerFast


VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "bengali-tokenizer": "tokenizer/data/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bengali-tokenizer": 512,
}


class AlbertBengaliTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Albert Bengali tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        tokenizer_file ():obj:`str`):
            path to the json file containing the fast tokenizer.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        tokenizer_file,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        padding_side="right",
        model_max_length=512,
        **kwargs,
    ):
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return super()._save_pretrained(
            file_names=(), save_directory=save_directory, filename_prefix=filename_prefix, legacy_format=False
        )
