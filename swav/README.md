# Training SWaV with decentralized averaging

This code trains [SwAV](https://arxiv.org/abs/2006.09882) model on [ImageNet](https://www.image-net.org/) using collaborative SGD. In our code we use [vissl](https://github.com/facebookresearch/vissl) and [ClassyVision](https://github.com/facebookresearch/ClassyVision) with some modifications.

## Requirements (for all participants):
* Install the library (`vissl`) from the root folder using [the guide](https://vissl.ai/tutorials/Installation) from source.
* Install the library (`ClassyVision`) from the root folder.
* Install hivemind (see [main README](../README.md)).

## How to run
1. Get ImageNet by following the [vissl guide](https://github.com/facebookresearch/vissl/blob/master/GETTING_STARTED.md).
2. Run the first DHT peer (aka "coordinator") on a node that is accessible to all trainers:
``` python run_initial_dht_node.py --listen_on [::]:1337 ```. After that, you can get INITIAL_DHT_ADDRESS and INITIAL_DHT_PORT from the stdout.
3. For all GPU trainers, run 
   
```
python vissl/tools/run_distributed_engines.py \
    hydra.verbose=true config=pretrain/swav/swav_1node_resnet_submit \
    config.CHECKPOINT.CHECKPOINT_ITER_FREQUENCY=30000 \
    +config.OPTIMIZER.batch_size_for_tracking=64 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=64  \
    +config.OPTIMIZER.lr=2.4 +config.OPTIMIZER.warmup_start_lr=0.3 \
    +config.OPTIMIZER.warmup_epochs=500 +config.OPTIMIZER.max_epochs=5000 \
    +config.OPTIMIZER.eta_min=0.0048 \
    +config.OPTIMIZER.exp_prefix="test_resnet50_swav_collaborative_experiment" \
    +config.OPTIMIZER.target_group_size=4 \
    +config.OPTIMIZER.max_allowed_epoch_difference=1 \
    +config.OPTIMIZER.total_steps_in_epoch=640 config.LOSS.swav_loss.queue.start_iter=98000 \
    +config.OPTIMIZER.report_progress_expiration=600 +config.DATA.TRAIN.DATA_PATHS=["$(IMAGENET_PATH)/train"] \
    config.OPTIMIZER.dht_listen_on_port=1124 config.OPTIMIZER.averager_listen_on_port=1125 \
    +config.OPTIMIZER.dht_initial_peers=["$(INITIAL_DHT_ADDRESS):$(INITIAL_DHT_PORT)"]
```
