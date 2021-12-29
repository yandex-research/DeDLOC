# Distributed Deep Learning in Open Collaborations

This repository contains the code for the NeurIPS 2021 paper

**"Distributed Deep Learning in Open Collaborations"**

*Michael Diskin\*, Alexey Bukhtiyarov\*, Max Ryabinin\*, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov,
Dmitry Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, Denis Mazur, Ilia Kobelev, Yacine
Jernite, Thomas Wolf, Gennady Pekhimenko*

Link: [ArXiv](https://arxiv.org/abs/2106.10207)

## Note

This repository contains a snapshot of the code used to conduct experiments in the paper.

Please use **[the up-to-date version](https://github.com/learning-at-home/hivemind)** of our library if you want to try out collaborative training and/or set up your own experiment. It contains many substantial improvements, including better documentation and fixed bugs.

## Installation

Before running the experiments, please set up the environment by following the steps below:

- Prepare an environment with python __3.7-3.9__. [Anaconda](https://www.anaconda.com/products/individual) is
  recommended, but not required
- Install the [hivemind](https://github.com/learning-at-home/hivemind) library from the master branch or by
  running `pip install hivemind==0.9.9.post1`

For all distributed experiments, the installation procedure must be repeated on every machine that participates in the
experiment. We recommend using machines with at least 2 CPU cores, 16 GB RAM and, when applicable, a low/mid-tier NVIDIA
GPU.

## Experiments

The code is divided into several sections matching the corresponding experiments:

- [`albert`](./albert) contains the code for controlled experiments with ALBERT-large on WikiText-103;
- [`swav`](./swav) is for training SwAV on ImageNet data;
- [`sahajbert`](./sahajbert) contains the code used to conduct a public collaborative experiment for the Bengali
  language ALBERT;
- [`p2p`](./p2p) is a step-by-step tutorial that explains decentralized NAT traversal and circuit relays.

We recommend running [`albert`](./albert) experiments first: other experiments build on top of its code and may reqire
more careful setup (e.g. for public participation). Furthermore, for this experiment, we
provide [a script](./albert/AWS_runner.ipynb) for launching experiments using preemptible GPUs in the cloud.

## Acknowledgements

This project is the result of a collaboration between
[Yandex](https://research.yandex.com/), [Hugging Face](https://huggingface.co/), [MIPT](https://mipt.ru/english/),
[HSE University](https://www.hse.ru/en/), [University of Toronto](https://www.utoronto.ca/),
[Vector Institute](https://vectorinstitute.ai/), and [Neuropark](https://neuropark.co/).

We also thank Stas Bekman, Dmitry Abulkhanov, Roman Zhytar, Alexander Ploshkin, Vsevolod Plokhotnyuk and Roman Kail for
their invaluable help with building the training infrastructure. Also, we thank Abhishek Thakur for helping with
downstream evaluation and Tanmoy Sarkar with Omar Sanseviero, who helped us organize the collaborative experiment and
gave regular status updates to the participants over the course of the training run.

## Contacts

Feel free to ask any questions in [our Discord chat](https://discord.gg/uGugx9zYvN) or [by email](mailto:mryabinin0@gmail.com).

## Citation

```bibtex
@inproceedings{diskin2021distributed,
    title = {Distributed Deep Learning In Open Collaborations},
    author = {Michael Diskin and Alexey Bukhtiyarov and Max Ryabinin and Lucile Saulnier and Quentin Lhoest and Anton Sinitsin and Dmitry Popov and Dmitriy Pyrkin and Maxim Kashirin and Alexander Borzunov and Albert Villanova del Moral and Denis Mazur and Ilia Kobelev and Yacine Jernite and Thomas Wolf and Gennady Pekhimenko},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year = {2021},
    url = {https://openreview.net/forum?id=FYHktcK-7v}
}
```
