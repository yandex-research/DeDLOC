# Training ALBERT with decentralized averaging

This tutorial will walk you through the steps to set up collaborative training with the ALBERT-large-v2 model and the WikiText-103 dataset. 
It uses Hugging Face [datasets](https://github.com/huggingface/datasets) and [transformers](https://github.com/huggingface/transformers/) libraries to compute local updates and `hivemind.CollaborativeOptimizer` to exchange information between peers.

### Preparation
* Install hivemind (see [main README](../README.md))
* Dependencies: `pip install -r requirements.txt`
* Preprocess data: `python tokenize_wikitext103.py`
* Run the coordinator 
```
 HIVEMIND_THREADS=128 python ./run_first_peer.py --dht_listen_on [::]:SOME_PORT  \
 --experiment_prefix SOME_NAME --wandb_project YOUR_PROJECT
```

The coordinator will then print ```Running DHT root at COORDINATOR_IP_HERE:COORDINATOR_PORT_HERE```,
you will need these values to launch additional peers.

* To start a GPU-enabled trainer, run
``` 
HIVEMIND_THREADS=128 python run_trainer.py \
--output_dir ./outputs_trainer --overwrite_output_dir  --logging_dir ./logs_trainer \
--logging_first_step --logging_steps 100   --initial_peers COORDINATOR_IP:COORDINATOR_PORT \
--experiment_prefix SOME_NAME --seed 42 --averaging_timeout 120  --bandwidth 1000 
 ```
Instead of COORDINATOR_IP:COORDINATOR_PORT, you can specify any existing trainers in the same format.

 
* To start an auxiliary CPU peer, run
``` 
HIVEMIND_THREADS=128 python run_aux.py  --output_dir ./outputs_aux \
--overwrite_output_dir   --logging_dir ./logs_aux --logging_first_step --logging_steps 100 \
--initial_peers COORDINATOR_IP:COORDINATOR_PORT   --experiment_prefix SOME_NAME --seed 42 \
--averaging_timeout 120 --fp16 False --bandwidth 1000
  ```
  
These peers do not contribute gradients, but assist others in gradient averaging.