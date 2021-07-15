# sahajBERT: A Collaboratively Trained Bengali Language Model

This section contains the code and instructions that were used when training
[sahajBERT](https://huggingface.co/neuropark/sahajBERT) in a collaborative setup.

## Preparation

* Install hivemind (see [main README](../README.md))
* Dependencies: `pip install -r requirements.txt`
* Preprocess the data by following the instructions in the `tokenizer_training` folder
* Upload the archive with preprocessed data to somewhere your peers can reach

## Running an experiment

Run the first DHT peer to welcome trainers and record training statistics (e.g. loss, performance):

- In this example, we use [wandb.ai](https://wandb.ai/site) to plot training metrics; If you're unfamiliar with Weights
  & Biases, here's a [tutorial](https://docs.wandb.ai/quickstart).
- Run `python run_first_peer.py --listen_on '[::]:*' --experiment_prefix NAME_YOUR_EXPERIMENT --wandb_project WANDB_PROJECT_HERE`
- `NAME_YOUR_EXPERIMENT` must be a unique name of this training run, e.g. `Bengali-albert`. It cannot contain `.` due to
  naming conventions.
- `WANDB_PROJECT_HERE` is a name of wandb project used to track training metrics. Multiple experiments can have the same
  project name.
- This peer will run a DHT node on a certain IP/port (`Running DHT root at ...`). You will need this address for next
  steps

To join the collaboration as a participant, run

``` 
HIVEMIND_THREADS=128 python run_trainer.py \
--output_dir ./outputs_trainer --overwrite_output_dir  --logging_dir ./logs_trainer \
--logging_first_step --logging_steps 100   --initial_peers COORDINATOR_IP:COORDINATOR_PORT \
--experiment_prefix NAME_YOUR_EXPERIMENT --seed 42 --averaging_timeout 120  --bandwidth 1000 
 ```

Instead of COORDINATOR_IP:COORDINATOR_PORT, you can specify any existing trainers in the same format.

### Collaborator notebook and authentication

We also provide an [example](./contributor_notebook.ipynb) of the Jupyter Notebook that was shared with the participants
as an easy way to join the experiment. Note that it is meant as a demonstration and not as a ready-to use solution; as a
result, it might require adaptation to the specifics of the experiment and the community that you intend to share it
with.

One possible solution for managing experiment access that was used for sahajBERT training is the authentication
mechanism. In [huggingface_auth.py](./huggingface_auth.py), you can see the client side of the authentication API that
can be used as an authorizer in Hivemind. For the server side, please refer to the
[collaborative-training-auth](https://github.com/huggingface/collaborative-training-auth) repo by Hugging Face.

## Downstream evaluation

We use scripts to evaluate on downstream tasks, based on the original finetuning scripts from the transformers
repository.

Datasets:

- NER: "wikiann", "bn"
- NCC: "indic_glue", "sna.bn"

Models:

- "neuropark/sahajBERT"
- "xlm-roberta-large"
- "ai4bharat/indic-bert"
- "neuralspace-reverie/indic-transformers-bn-roberta"

Required arguments:

```shell
--model_name_or_path
--output_dir
```

### Examples

NER

```shell
python train_ner.py \
  --model_name_or_path xlm-roberta-large \
  --output_dir sahajbert/ner \
  --learning_rate 1e-5 \
  --max_seq_length 128 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --early_stopping_patience 3 \
  --early_stopping_threshold 0.01
```

NCC

```shell
python train_ncc.py \
  --model_name_or_path xlm-roberta-large \
  --output_dir sahajbert/ncc \
  --learning_rate 1e-5 \
  --max_seq_length 128 \
  --num_train_epochs 20 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --early_stopping_patience 3 \
  --early_stopping_threshold 0.01
```
