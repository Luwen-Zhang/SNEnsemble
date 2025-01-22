# SNEnsemble

It is the repository for the article (**Update after acceptance**).

## Requirements

This work is built on our developed open-source tabular prediction benchmark platform `tabular_ensemble`. 

Note that the requirements change from time to time in `tabular_ensemble`. To reproduce our dependencies:

* Install python 3.10.12. 
* Install `torch==1.13.1` following [pytorch's guide](https://pytorch.org/get-started/previous-versions/). We use CUDA 11.6 for our work.

* Install requirements

```shell
pip install -r requirements.txt
```

* Run  `pip install tabular_ensemble==0.2`. Most of its dependencies are satisfied.

## Usage

* Running experiments

The bash script `run.sh` lists the commands to run experiments using `main.py`, `main_clustering.py`, `main_for_analysis.py`, and `main_layup.py`. Note that each command in `run.sh` may cost several days to excecute (1-4 days in our environment). They are both CPU and GPU consuming.

* Analyze results

The notebook `analysis.ipynb` depicts codes for our analysis (mostly visualizations).

To analyze our results, you should download checkpoint files (around 25 GB) from [this link](https://doi.org/10.5281/zenodo.13858982). Don't hesitate to contact us if the link is invalid. One can check the `analysis.ipynb` to see what folders listed in the following directory tree (under `output`) are needed for specific analysis. 

## Directory tree

Here is the file tree of this repository and explanations of each folder/file. Under the output folder, comments of each subfolder indicate specific commands in `run.sh` used to generate these checkpoints and information.

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
│   │   ├── 2024-07-16-07-50-38-0_composite B316 mcd	# main.py
│   │   ├── 2024-04-12-19-04-14-0_composite A316 mcd	# main.py
│   │   ├── 2024-04-12-21-32-11-0_composite C622 mcd	# main.py
│   │   ├── 2024-07-14-00-05-14-0_composite B622 mcd	# main.py
│   │   ├── 2024-04-16-22-25-12-0_composite clusters A622	# main_clustering.py
│   │   ├── 2024-07-14-00-08-13-0_composite clusters B622	# main_clustering.py
│   │   ├── 2024-04-16-22-37-18-0_composite-I1 clusters C622	# main_clustering.py
│   │   ├── 2024-04-16-23-10-09-0_composite A811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-04-17-09-22-24-0_composite C811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-07-15-21-31-48-0_composite B811 mcd analysis	# main_for_analysis.py
│   │   ├── 2024-04-17-09-55-16-0_composite_no_relative_stress A622 # main_no_relative_stress.py
│   │   ├── 2024-07-16-04-07-45-0_composite_no_relative_stress B622 # main_no_relative_stress.py
│   │   ├── 2024-04-20-16-02-23-0_composite_no_relative_stress C622 # main_no_relative_stress.py
│   │   ├── 2024-04-29-21-01-40-0_composite-I2 nowrap A622 		# main.py --nowrap
│   │   ├── 2024-07-14-00-08-06-0_composite nowrap B622 		# main.py --nowrap
│   │   ├── 2024-04-29-21-01-40-0_composite-I1 nowrap C622 		# main.py --nowrap
│   │   ├── 2024-06-25-23-49-03-0_composite-I1 A622 useraw		# main.py --use_raw
│   │   ├── 2024-07-17-01-01-03-0_composite B622 useraw			# main.py --use_raw
│   │   └── 2024-06-25-23-49-03-0_composite-I1-I1 C622 useraw	# main.py --use_raw
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

We run our experiments on Siyuan-1 cluster supported by the Center for High Performance Computing at Shanghai Jiao Tong University (with Nvidia A100 40GB GPU), or one RTX 3090 24GB GPU. The environment and configurations of each experiment can be seen in `log.txt` in each output folder. Results might differ on different devices/environments due to minor differences caused by devices and Bayesian hyperparameter optimizations, but our conclusions are identical among devices. 

Results are analyzed on a personal computer with Ubuntu 18.04, Intel Core i9-11900K, Nvidia RTX 3090. 

## Fatigue dataset

We reformat and merge fatigue testing records from SNL/MSU/DOE, OptiMat, UPWIND, and FACT databases into a single database with careful considerations, including filtering invalid entries, rename variables, fix improper calculations in the records, to obtain clean and consistent data to fit machine learning usages. 
Scripts we use to process databases and obtain the merged dataset are all included in the `data` folder.  

In `data/merge_data/mlfatigue_data` , `SNL_MSU_DOE_raw.xlsx` and `FACT_raw.xlsx` are manually copied from original files of OptiDat and SNL/MSU/DOE. `Upwind_combine.xlsx` and `OptiMat_combine.xlsx` are obtained using another tool named `Tables2Table` contained in `data/Tables2Table` . In `data/Tables2Table`, `OptiMat.xlsx` and `Upwind.xlsx` are manually copied from original files of OptiDat. In both these two folders, the `src` subfolder contains corresponding scripts to process datasets. 

Units of features are unified; entries of numerical features that contain characters that are not units are considered as empty; names of categories of categorical features are made consistent; a column named `Material_Code` representing different laminates (See definitions in the following paragraph) is generated to distinguish different resin material, fiber material, lay-up, specimen geometry, etc. and to match fatigue data and static test data of each laminate; entries of some correlated features, especially minimum stress, maximum stress, and R-value, are filled if missing using their relationships and are corrected if conflicts exist such as a positive maximum stress with an R-value greater than 1; lay-up information recorded in a specialized terminology in the composite laminate literature such as $[[0/\pm45]_\mathrm{S}/90]_\mathrm{S}$ is translated in a sequential form using a developed recursive algorithm. 

The column named `Material_Code` that distinguish different laminates is defined as followed: For the SNL/MSU/DOE dataset, laminates are distinguished by the combination of two columns, namely "Material" (a general identification of lay-up, fibers, and resin) and "Lay-up". For the OptiMat dataset (see our released repository), laminates are defined by three columns ("Plate", "Geometry", and "Laminate"), all of which are identifications of fibers, resin, specimen, and lay-up sequences. For the Upwind dataset (see our released repository), they are defined by two columns ("plate" and "geometry"), both of which are identification codes defined internally in the dataset. For the FACT dataset (see our released repository), they are defined by "material" (resin type) and "laminate" (lay-up). 

We emphasize that both strain-controlled and stress-controlled data points are included and cannot be clearly distinguished, which is a deficiency of this dataset. SNL/MSU/DOE and Upwind did not clearly record the control mode for every data point. A total of 68 certainly strain-controlled data points in FACT and OptiMat databases also recorded corresponding stress levels. Additionally, waveform recorded only in a small portion of points is also ambiguous in databases. These two categorical features, i.e., waveform selected from \{Sinusoidal, Triangular\} and control mode selected from \{Load, Displacement\}, are not used due to a high absence ratio but are recorded in the released data file. 

We leave the original file of OptiDat (containing the latter three datasets) in the folder. For the original file of SNL/MSU/DOE, please visit [this link](https://energy.sandia.gov/programs/renewable-energy/wind-power/rotor-innovation/rotor-reliability/mhk-materials-database/). Any derived database should be directly based on the original files instead of our merged database. We are not responsible for the accuracy of the testing records.

We acknowledge Montana State University and Mr. Sibrand Raijmaekers for the databases.

## Citation

If you find our work useful in your research, please consider citing us as (**Update after acceptance**)


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

