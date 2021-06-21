import datasets

from tokenizer_model import SentencePieceUnigramCustomizedTokenizer


vocab_size = 31_995
input_sentence_size = None

# Initialize a dataset
dataset = datasets.load_dataset("oscar", name="unshuffled_deduplicated_bn", split="train")

tokenizer = SentencePieceUnigramCustomizedTokenizer()

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i : i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
    special_tokens=["<pad>", "<unk>", "[CLS]", "[SEP]", "[MASK]"],
)

tokenizer.save(f"data/tokenizer.json")
