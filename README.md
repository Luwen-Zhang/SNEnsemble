# Machine-learning-based Fatigue Life Prediction

It will be published as a public repository after acceptance.

This repository aims to build a universal and extensible benchmark platform for predicting fatigue useful life using tabular data. We implement our own model and compare its performance with baseline modelbases. Though our paper aims at fibre reinforced composites, we believe that the repository is valuable for other types of composites, materials, or even tabular datasets from other research fields.

## Usage

Bash scripts (e.g. `run.sh` and `run_thiswork.sh`) gives several examples to use the scripts (e.g. `main.py` and `main_thiswork.py`). It contains command lines to reproduce results in our paper.

## Requirements

First, install `torch==1.12.0` with CUDA 1.16 (if a Nvidia GPU is available). 

```shell
pip install torch==1.12.0+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
```

Then install dependencies

```
pip install -r requirement.txt
```

## Our environment

We run our experiments on a personal computer with Ubuntu 18.04, Intel Core i9-11900K, Nvidia RTX 3090. Results might differ on different devices/environments. We have verified main conclusions on Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University (with Nvidia A100 GPU). The environment for each trial can be seen in `log.txt`.

## Modelbases

The respository merges following well-established modelbases as baselines:

* [AutoGluon](https://github.com/autogluon/autogluon)

* [WideDeep](https://github.com/jrzaurin/pytorch-widedeep)

* [Pytorch-Tabular](https://github.com/manujosephv/pytorch_tabular)


## Implementing new features

New modelbases, individual models, tabular databases, feature derivation procedures, data preprocessing procedures, etc., can be easily extended in the framework. New features follow certain structures given by parent classes, and several methods should be implemented for them. See `src.model` for details.

## Contribution

Feel free to create issues if you implement new features, find mistakes, or have any question.

## Acknowledgment

We receive valuable supports from Mr. Sibrand Raijmaekers (LM Wind Power) for Upwind, OptiMat, and FACT databases. We also feel grateful for the open-sourced database SNL/MSU/DOE from Sandia National Laboratories.
