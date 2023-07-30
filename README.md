# Machine Learning Benchmark Platform for Fatigue Life Prediction

It will be published as a public repository after acceptance.

## Usage

Bash scripts (e.g. `run.sh` and `run_thiswork.sh`) gives several examples to use the scripts (e.g. `main.py` and `main_thiswork.py`). It contains command lines to reproduce results in our paper.

## Requirements

This work is built on our developing open-source tabular prediction benchmark platform [`tabular_ensemble`](https://github.com/LuoXueling/tabular_ensemble). 

Note that the requirements change from time to time in `tabular_ensemble`. To reproduce our dependencies:

* Before installing  `tabular_ensemble` , install `torch==1.12.0` with CUDA 1.16 (if a Nvidia GPU is available)

```shell
pip install torch==1.12.0+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
```

* Install requirements

```shell
pip install -r requirements.txt
```

* Install  `tabular_ensemble`. See the installation instruction there.

## Our environment

We run our experiments on a personal computer with Ubuntu 18.04, Intel Core i9-11900K, Nvidia RTX 3090. Results might differ on different devices/environments. We have verified main conclusions on Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University (with Nvidia A100 GPU). The environment and configurations for each trial can be seen in `log.txt`.

