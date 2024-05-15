# ENSEMBLE

It is the repository for the article *ENSEMBLE: Ensembling empirical models and machine learning exemplified by fatigue life extrapolation of fiber-reinforced composites* (**Update after acceptance**).

## Requirements

This work is built on our developed open-source tabular prediction benchmark platform [`tabular_ensemble`](https://github.com/LuoXueling/tabular_ensemble). 

Note that the requirements change from time to time in `tabular_ensemble`. To reproduce our dependencies:

* Install python 3.10.12. 
* Install `torch==1.13.1` following [pytorch's guide](https://pytorch.org/get-started/previous-versions/). We use CUDA 11.6 for our work.

* Install requirements

```shell
pip install -r requirements.txt
```

* Install  [`tabular_ensemble`](https://github.com/LuoXueling/tabular_ensemble). Most of its dependencies are satisfied.

## Usage

* Running experiments

Bash scripts `run.sh` gives the commands to run experiments using `main.py`, `main_clustering.py`, `main_for_analysis.py`, and `main_layup.py`. Note that each command in `run.sh` may cost several days to excecute (1-4 days in our environment). They are both CPU and GPU consuming.

* Analyze results

The notebook `analysis.ipynb` depicts codes for our analysis (mostly visualizations).

To analyze our results, you should download checkpoint files (around 50 GB) from this link (**Update after acceptance**). Don't hesitate to contact us if the link is invalid.

## Directory tree

Here is the file tree of this repository and explanations of each folder/file.

```
.
├── analysis.ipynb 						# Analyze our results
├── calculate_shap.py 					# A supplementary file for analysis.ipynb
├── configs 							# Configuration files in this work
│   ├── composite_no_relative_stress.py	# Compare results without derived features
│   ├── composite.py					# Main configuration
│   └── modulus.py						# Ultilize lay-up information
├── continue_previous.py				# Continue accidentally terminated tasks.
├── data								# Data files and descriptions.
│   ├── merge_data						# Scripts to merge four databases
│   ├── Tables2Table					# Scripts to pre-process Upwind and OptiMat
│   ├── 2019_SNL_MSU_DOE_Composite-Materials-Database_Wind_29pt0.xlsx
│   ├── composite_database_03222024.csv
│   ├── composite_database_03222024_with_extended_information.xlsx
│   ├── composite_database_layup_modulus_03302024.csv
│   ├── info_03222024.csv
│   └── Optidat UPWIND 28_3_2017.xls
├── main_clustering.py		# Compare different clustering methods in the framework
├── main_for_analysis.py	# Run experiments once with a 8:1:1 ratio
├── main_layup.py			# Ultilize lay-up information
├── main_no_relative_stress.py			# Compare results without derived features
├── main.py								# Main script for most experiments
├── output								# Checkpoint folders you might download
│   ├── analyse							# Results of analysis
│   │   └── paper_plots
│   ├── composite_database_03222024		# Results of experiments
│   │   ├── 2024-04-12-09-34-02-0_composite C316 mcd 	# main.py
│   │   ├── 2024-04-12-19-02-45-0_composite A622 mcd	# main.py
│   │   ├── 2024-04-12-19-03-01-0_composite B316 mcd	# main.py
│   │   ├── 2024-04-12-19-04-14-0_composite A316 mcd	# main.py
│   │   ├── 2024-04-12-21-32-11-0_composite C622 mcd	# main.py
│   │   ├── 2024-04-12-21-32-11-0_composite-I1 B622 mcd	# main.py
│   │   ├── 2024-04-16-22-25-12-0_composite clusters A622	# main_clustering.py
│   │   ├── 2024-04-16-22-37-18-0_composite clusters B622	# main_clustering.py
│   │   ├── 2024-04-16-22-37-18-0_composite-I1 clusters C622	# main_clustering.py
│   │   ├── 2024-04-16-23-10-09-0_composite A811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-04-17-09-22-24-0_composite C811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-04-20-20-06-36-0_composite B811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-04-17-09-55-16-0_composite_no_relative_stress A622 # main_no_relative_stress.py
│   │   ├── 2024-04-19-04-09-12-0_composite_no_relative_stress B622 # main_no_relative_stress.py
│   │   ├── 2024-04-20-16-02-23-0_composite_no_relative_stress C622 # main_no_relative_stress.py
│   │   ├── 2024-04-29-21-01-40-0_composite-I2 nowrap A622 		# main.py --nowrap
│   │   ├── 2024-04-29-21-01-40-0_composite nowrap B622 		# main.py --nowrap
│   │   └── 2024-04-29-21-01-40-0_composite-I1 nowrap C622 		# main.py --nowrap
│   └── composite_database_layup_modulus_03302024	# main_layup.py
│       └── 2024-04-03-22-49-46-0_modulus
├── requirements.txt	# Frozen dependencies
├── run.sh				# Commands to run experiments
└── src					# Scripts of this work
    ├── data
    │   ├── dataderiver.py
    │   ├── datasplitter.py
    │   └── __init__.py
    ├── __init__.py
    ├── model
    │   ├── __init__.py
    │   ├── _thiswork	# Main scripts of the proposed method
    │   │   ├── bayes_nn	# The Bayesian neural network (MCDropout and others)
    │   │   ├── clustering	# The empirical models and general clustering algorithms
    │   │   ├── __init__.py
    │   │   ├── models_clustering.py # The combination of modules
    │   │   ├── pca			# Principle component analysis
    │   ├── _thiswork_layup	# The Transformer model for lay-up utilization
    │   ├── thiswork_layup.py	# Defines the Transformer model
    │   └── thiswork.py		# Defines the proposed methods
    └── trainer				# Defines some useful functions
        ├── __init__.py
        └── trainer.py
```

## Our environment

We run our experiments on Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University (with Nvidia A100 40GB GPU). The environment and configurations of each experiment can be seen in `log.txt` in each output folder. Results might differ on different devices/environments due to minor differences caused by devices and major differences caused by Bayesian hyperparameter optimizations. 

Results are analyzed on a personal computer with Ubuntu 18.04, Intel Core i9-11900K, Nvidia RTX 3090. 

## Merging data

Scripts we use to process databases and obtain the merged dataset are all included in the `data` folder.  

In `data/merge_data/mlfatigue_data` , `SNL_MSU_DOE_raw.xlsx` and `FACT_raw.xlsx` are manually copied from original files of OptiDat and SNL/MSU/DOE. `Upwind_combine.xlsx` and `OptiMat_combine.xlsx` are obtained using another released tool named [`Tables2Table`](https://github.com/LuoXueling/Tables2Table), which is also contained in `data/Tables2Table`  for convenience. In `data/Tables2Table`, `OptiMat.xlsx` and `Upwind.xlsx` are manually copied from original files of OptiDat. In both these two folders, the `src` subfolder contains corresponding scripts to process datasets. 

## Citation

If you find our work useful in your research, please consider citing us as (**Update after acceptance**)

## Data copyright

We reformat and merge fatigue testing records from SNL/MSU/DOE, OptiMat, UPWIND, and FACT databases into a single database with careful considerations, including filtering invalid entries, rename variables, fix improper calculations in the records, to obtain clean and consistent data to fit machine learning usages. 

We leave the original file of OptiDat (containing the latter three datasets) in the folder. For the original file of SNL/MSU/DOE, please visit [this link](https://energy.sandia.gov/programs/renewable-energy/wind-power/rotor-innovation/rotor-reliability/mhk-materials-database/). Any derived database should be directly based on the original files instead of our merged database. We are not responsible for the accuracy of the testing records.

We acknowledge Montana State University and Mr. Sibrand Raijmaekers for the databases.

Following are copyright contents from the original databases.

### SNL/MSU/DOE database

```
SNL/MSU/DOE COMPOSITE MATERIAL FATIGUE DATABASE
Mechanical Properties of Composite Materials for Wind Turbine Blades
Version 29.0; May 13, 2019
Montana State University - Bozeman

This database was prepared as a part of  work sponsored by an agency of the U.S. Government.
Neither the U.S. Government, nor any agency thereof, nor any of  their employees, nor any of their contractors,
subcontractors, or their employees, makes any warranty, expressed or implied, or assumes any legal liability or
responsibility for the accuracy, completeness, or usefulness of this program, or represents that opinion expressed
herein do not necessarily state or reflect those of the U.S. Government, any agency thereof or any of their 
contractors or subcontractors. The material presented in this publication should not be used or relied upon for 
any specific application without competent examination and verification of its accuracy, suitability, and 
applicability by qualified professionals. Reference herein to any specific commercial product or process 
by trade name, trademark, manufacturer, or otherwise, does not necessarily constitute or imply its endorsement 
or recommendation. This 29.0 version of the database supersedes all previous versions.
This database is maintained by Daniel Samborsky (DanielS@montana.edu, 406-994-7186) at Montana State University
Updates, reports and conference papers can be downloaded from www.montana.edu/composites or windpower.sandia.gov/materials-reliability
```

### Optidat UPWIND database

```
Copyright notice:

This database is Copyright (C) 2007 by the Knowledge Centre Wind turbine Materials and Constructions (KC-WMC).

Permission is granted to use this work in the following ways:

      1) You may make private copies for your own personal use.
      2) You may create derivative works for your own personal use.
      3) You may publish short excerpts from the database, provided that a reference is included to: "Nijssen, R.P.L., ‘OptiDAT – fatigue of wind turbine materials database’, regular updates via www.kc-wmc.nl"
      4) You may distribute this work or make it available for copying by
         others only if ALL of the following three conditions are met:
         a) The information content of the database is unchanged.
            (You may not add, delete, or modify records.  You may, however, reformat and/or reorder the data.)
         b) The distribution is not made for monetary or material gain.
         c) A copy of this notice is included with every distributed copy of the database.

Disclaimer:  The database is supplied without any warranty, stated or implied. In particular, no claim is made that it is accurate, complete, or suitable for any purpose.  Use it at your own risk.
```

