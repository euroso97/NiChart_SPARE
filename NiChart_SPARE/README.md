# Package Design
## Core modules
### __init__.py
* entry point configuration

### __main__.py
* handle input arguments
* train/inference verification & function calls for different ML approaches (currently SVM, MLP (TBA))

### data_prep.py
* functions for ROI data preprocessing
	* for classification
	* for regression
* feature encoding
* ROI residualization for Age, Sex, DLICV effect.

| Type | function name |
| -------- | ------- |
| ROI data preprocessing | load_csv_data, validate_dataframe, encode_feature_df, preprocess_classification_data, preprocess_regression_data,  |
| Age, Sex, ICV linear effect removal | apply_cvm_residualization |


### utils.py
* Misc utility functions used across all pipelines
* checking functions:
	* if regression model
	* if valid dataframe
* load csv data, drop certain columns

| Type | function name |
| -------- | ------- |
| Hyperparameter | expspace |
| verify task type | is_regression_model |

### trainer.py (bygone) -> DEPRECIATED

### svm.py
| Type | function name |
| -------- | ------- |
| SVM model specific functions | get_pipeline_module, get_svm_hyperparameter_grids |
| SVM training | train_svm_model |
| SVM inference | predict_svm_model |
| SVM save & load | create_svm_training_info, save_svm_model, load_svm_model |


### mlp.py (TBA)
* MLP model specific functions

| Type | function name |
| -------- | ------- |
|  |  |

------------------

## Pipelines
### # General
#### SPARE-SVM-Classifcation: spare_svm_classification.py
* Standard function for SVM based classification

| Type | function name |
| -------- | ------- |
|  |  |

#### SPARE-SVM-Regression: spare_svm_regression.py
* Standard functions for SVM based regression

| Type | function name |
| -------- | ------- |
|  |  |

### # Specialized
#### SPARE-BA (Brain Age): spare_ba.py
* SPARE-BA specific regression
* Includes final SVM correction

#### SPARE-AD (Alzheimer's disease): spare_ad.py
* SPARE-AD specific classification
* Amyloid positive, negative (TBA)

#### SPARE-CVMs: HT, HL, T2D, OB, SM
* SPARE-CVMs 
* optional Age, Sex, (DL)ICV effect removal prior to classification