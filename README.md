# Machine-learning-based Fatigue Life Prediction

It will be published as a public repository after acceptance.

This repository aims to build a universal and extensible benchmark platform for predicting fatigue useful life using tabular data. We implement our own model and compare its performance with baseline modelbases. Though our paper aims at fibre reinforced composites, we believe that the repository is valuable for other types of composites, materials, or even tabular datasets from other research fields.

## Usage

`run.sh` gives several examples to use the scripts. It contains command lines to reproduce results in our paper. Notebooks under the `notebook` directory provides implemented visualization methods.

## Requirements

See `requirement.txt` for details. Besides, we assume that `torch` is already installed and it's not included in `requirement.txt` because customized installation is prefered. For reproducibility, we use `torch==1.10.2+cu102` and `python 3.7` on cpu to produce our results.

## Our environment

To produce our results presented in the paper, we run our experiments on a personal computer with Ubuntu 18.04 (Intel Core i9-11900K) and computing servers with DGX2 (π 2.0 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University). Results might differ on different devices/environments.

## Modelbases

The respository merges following well-established modelbases as baselines:

* [AutoGluon](https://github.com/autogluon/autogluon)
* [Pytorch-Tabular](https://github.com/manujosephv/pytorch_tabular)

## Implementing new features

New modelbases, individual models, tabular databases, feature derivation procedures, data preprocessing procedures, etc., can be easily extended in the framework. New features follow certain structures given by parent classes, and several methods should be implemented for them.

### New modelbases

* Parent class: `AbstractModel`
* Methods to be implemented: `_train`, `_predict`, `_get_model_names`
* Examples: `src/core/model.py`

### New models

* Parent class: `AbstractModel` or `TorchModel` (Pytorch based models should be implemented as subclasses of `TorchModel` which implements necessary methods.)
* Methods to be implemented: `_train`, `_predict`, `_get_model_names` (For `TorchModel`, only  `_new_model` and `_get_model_names` are required)
* Examples: `src/core/model.TabNet`, `src/core/model.MLP`

### New databases

* Examples: `data/*.xlsx`
* Requirements: 
  * A column named `Material_Code` is required for identifying different materials.
  * A corresponding configuration file should be created, following the format of examples given in the `configs` repository. All parameters should follow the structure of `configs/base_config.py`, otherwise errors will be raised when loading datasets.
* Remarks when implementing: 
  * We recommand column names that are consistent with those in given datasets so that features could be identified easily in our new powerful model. 

### New feature derivation procedures

* Parent class: `AbstractDeriver`
* Methods to be implemented: `derive`
* Remarks when implementing: 
  * One can add new features during data loading by adding items (the name of the class) to `data_derivers` in configuration files. Arguments passed to `derive` are given by a `dict` in the configuration.
  * `'stacked': True` means add the feature(s) as tabular features, otherwise they will be treated as individual matrices and will be passed to `TorchModel`s using the `additional_tensors` argument.
* Examples: `src/core/dataderiver.py`

### New data preprocessing procedures

* Parent class: `AbstractDeriver` (only change the number of data points like removing outliers) or `AbstractTransformer`  (change values of features like missing value imputation or standard scaling)
* Methods to be implemented: `fit_transform`, `transform`.
* Remarks when implementing:
  * In `fit_transform`, the property `record_features` stores Trainer.feature_names`. Therefore, when calling `transform`, the `DataFrame` with `record_features` and `Trainer.label_name` will be returned.
  * One can add new data processing steps by adding items to `data_processors` in configuration files.
* Examples: `src/core/dataprocessor.py`

### New data split methods

* Parent class: `AbstractSplitter`
* Methods to be implemented: `_split`
* Remarks when implementing:
  * The returned indices should not have intersection with each other.
  * The returned indices are one-dimensional `numpy.ndarray`.
* Examples: `src/core/datasplitter.py`

## Contribution

Feel free to make pull requests if you implement new features or find mistakes.

## Acknowledgment

We receive valuable supports from Mr. Sibrand Raijmaekers (LM Wind Power) for Upwind, OptiMat, and FACT databases. We also feel grateful for the open-sourced database SNL/MSU/DOE from Sandia National Laboratories.

## Citation

If you use this repository in your work, please cite us as:

```
FILLED AFTER PUBLICATION
```

