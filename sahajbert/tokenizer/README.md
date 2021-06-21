# Tokenizer training

To reproduce the tokenizer training, you can install the necessary dependencies in the virtual environment of your choice with the following command:
```
pip install -r requirements.txt
```
Then, launch the training of the tokenizer with the following command:
```
python tokenizer_training_custom.py 
```

In order to load the tokenizer with the transformers library, we need to do some manuel changes:

1. Put the `tokenizer.json` file and the `special_tokens_map.json` and `tokenizer_config.json` files stored in the `data` folder in a same new folder with the name of the tokenizer, for example `sahajBERT-tokenizer`
2. Make 2 manual modifications of the `tokenizer.json` file:

    a. Change the value `{"id": 4, "special": true, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}` in the `added_tokens` list into `{"id": 4, "special": true, "content": "[MASK]", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false}`
   
    b. Change the value associated with the key `unk_id` in the dictionary associated to the key `model` into `1`

Thanks to these last changes, the tokenizer can be loaded in the following way:
```
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("sahajBERT-tokenizer")
```