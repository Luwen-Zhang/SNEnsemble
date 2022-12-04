# Machine-learning-based Fatigue Life Prediction

It will be published as a public repository after acceptance.

This repository aims to build a universal and extensible benchmark platform for predicting fatigue useful life using tabular data. We implement our own model and compare its performance with baseline modelbases. Though our paper aims at fibre reinforced composites, we believe that the repository is valuable for other types of composites, materials, or even tabular datasets from other research fields.

## Usage

`run.sh` gives several examples to use the scripts. It contains command lines to reproduce results in our paper. Notebooks under the `notebook` directory provides implemented visualization methods.

## Requirements

See `requirement.txt` for details. Besides, we assume that `torch` is already installed and it's not included in `requirement.txt` because customized installation is prefered. For reproducibility, we use `torch==1.10.2+cu102` and `python 3.7` on cpu to produce our results.

## Modelbases

The respository merges following well-established modelbases as baselines:

* [AutoGluon](https://github.com/autogluon/autogluon)
* [Pytorch-Tabular](https://github.com/manujosephv/pytorch_tabular)

## Performance



## Implementing new features

New modelbases, individual models, tabular databases, feature derivation procedures and data preprocessing procedures can be easily extended in the framework.

### New modelbases

New modelbases can be introduced by implementing subclasses of `AbstractModel`. For each modelbase, the methods `_train()`, `_predict()`, and `_get_model_names` should be implemented and the property `program` should be specified. See   `src/core/model` for examples.

### New models

New models can be regarded as modelbases with only one model (`src/core/model.TabNet` for example). Note that Pytorch based models should be implemented as subclasses of `TorchModel` which implements necessary methods and only `_new_model` and `_get_model_names` are required.

### New databases

We include datasets in the `data` repository in `.xlsx` format. For each dataset, a column named `Material_Code` is required for identifying different materials. We recommand column names that are consistent with those in given datasets so that features could be identified easily in our new powerful model. 

A corresponding configuration file should be created, following the format of examples given in the `configs` repository. All parameters should follow the structure of `configs/base_config.py`, otherwise errors will be raised when loading datasets.

### New feature derivation procedures

Implementing subclasses of `AbstractDeriver` and adding them to `deriver_mapping` in `src/core/dataderiver` , one can add new features during data loading by adding items to `data_derivers` in configuration files. New procedure should implement the `derive` method. Parameters of the `derive` method will be passed using the dict in the configuration. Note that `'stacked': True` means add the feature(s) as tabular features, otherwise they will be treated as individual matrices and will be passed to `TorchModel`s using the `additional_tensors` argument. We have given examples in `src/core/dataderiver.py`.

### New data preprocessing procedures

We divide preprocessing procedures into two cataglories: `AbstractProcessor` and `AbstractTransformer`. The former ones will only change the number of data points (like removing outliers), but the latter ones change values of features (like missing value imputation or standard scaling). We give examples in `src/core/dataprocessor.py`. New procedures should implement `fit_transform` and `transform` methods and should be added to `processor_mapping`, therefore one can add new data processing steps by adding items to `data_processors` in configuration files.

## Contribution

Feel free to make pull requests if you implement new features or find mistakes.

## Acknowledgment

We receive valuable supports from Mr. Sibrand Raijmaekers (LM Wind Power) for Upwind, OptiMat, and FACT databases. We also feel grateful for the open-sourced database SNL/MSU/DOE from Sandia National Laboratories.

## Citation

If you use this repository in your work, please cite us as:

```
FILLED AFTER PUBLICATION
```

