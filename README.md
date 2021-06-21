# Distributed Deep Learning in Open Collaborations

This repository contains the code for the paper

**«Distributed Deep Learning in Open Collaborations»**

*Michael Diskin\*, Alexey Bukhtiyarov\*, Max Ryabinin\*, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitry Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf, Gennady Pekhimenko*

Link: [ArXiv](https://arxiv.org/abs/2106.10207)

## Installation

Before running the experiments, please set up the environment by following the steps below:

- Prepare an environment with python __3.7-3.9__. [Anaconda](https://www.anaconda.com/products/individual) is recommended, but not required
- Ensure that your machine has a recent version of Golang (1.15 or higher). To install Go, follow the [instructions](https://golang.org/doc/install) on the official website.
- Install the [hivemind](https://github.com/learning-at-home/hivemind) library from master or by running `pip install hivemind==0.9.9`

For all distributed experiments, the installation procedure must be repeated on every machine that participates in the
experiment. We recommend using machines with at least 2 CPU cores, 16 GB RAM and, when applicable, a low/mid-tier NVIDIA
GPU.

## Experiments

The code is divided into several sections matching the corresponding experiments:

- [`albert`](./albert) contains the code for controlled experiments with ALBERT-large on WikiText-103;
- [`swav`](./swav) is for training SwAV on ImageNet data;
- [`sahajbert`](./sahajbert) contains the code used to conduct a public collaborative experiment for the Bengali language ALBERT;
- [`p2p`](./p2p) is a step-by-step tutorial that explains decentralized NAT traversal and circuit relays.

We recommend running [`albert`](./albert) experiments first: other experiments build on top of its code and may
reqire more careful setup (e.g. for public participation). Furthermore, for this experiment, we
provide [a script](./albert/AWS_runner.ipynb) for launching experiments using preemptible GPUs in the cloud.


## Citation:
```
@misc{diskin2021distributed,
      title={Distributed Deep Learning in Open Collaborations}, 
      author={Michael Diskin and 
              Alexey Bukhtiyarov and 
              Max Ryabinin and 
              Lucile Saulnier and 
              Quentin Lhoest and 
              Anton Sinitsin and 
              Dmitry Popov and 
              Dmitry Pyrkin and 
              Maxim Kashirin and 
              Alexander Borzunov and 
              Albert Villanova del Moral and 
              Denis Mazur and 
              Ilia Kobelev and 
              Yacine Jernite and 
              Thomas Wolf and 
              Gennady Pekhimenko},
      year={2021},
      eprint={2106.10207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```