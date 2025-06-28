# Integrating car types in a car ownership model

The car ownership model estimates actual car ownership of households, including the number of cars owned and their types, based on their socioeconomic and mobility-related characteristics.
In the following, the model is applied to the Paris region for which trained classifiers are provided.

## Input data
  
The model requires two input data to be added to the `data/` folder (see the further description below):
(1) a household transport survey (here the EGT 2020 survey, which is not provided for confidentiality purposes) and
(2) origin-destination statistics for professional mobility (here the French national census, MOBPRO dataset)

Input datasets: 
* `data/Recensement_MOBPRO_2018.csv` is the public census dataset for professional mobility, available [online](https://www.insee.fr/fr/statistiques/5395749?sommaire=5395764&q=mobilit%C3%A9s+professionnelles+2018)
* `data/Varmod_MOBPRO_2018.csv` contains the variable descriptions for this dataset (available online via the same link)
* `data/EGT_2020/` contains the EGT 2018-2020 mobility survey dataset (*)

## Get started: data preprocessing

Install the environment.
In the `src/input/` folder, we load the input data and generate summary statistics on it, as well as processing the training datasets.

First, run the following scripts to load and clean the input datasets: `src/input/load_data_accessibility_FR.py` and `src/input/clean_municipalities.py`.
You can generate statistics for the household mobility survey with the command `src/input/generate_statistics_EGT.py`.

Then run `src/input/preprocess_data_EGT.py` to generate training datasets.

The output datasets are stored in `training_datasets/` (not provided here) and are then used to train the model:
* `training_datasets/Nvehicles.csv`
* `training_datasets/Population.csv`
* `training_datasets/Vehicle_types.csv`

Preprocessing codes description summary:
* `src/input/load_data_accessibility_FR.py` - loads data from the public census and calculates the regional accessibility variable (car and public transport modal shares per municipality)
* `src/input/generate_statistics_EGT.py` - generates statistics from EGT 2018-2020
* `src/input/clean_municipalities.py` - data cleaning for EGT work / study municipalities
* `src/input/preprocess_data_EGT.py` - generates training datasets

## Model training

We train, save and assess one model per predicted variable with EGT data running `src/validation/model_training_validation.py`.
14 ML and DCM models are assessed and compared based on the MCC, then the best-performing model is trained on the entire dataset and saved.
Model validation includes the confusion matrix, the area under the curve (AUC), logistic regression coefficients and feature importance for this model.
Finally, the trained model is saved as pickle file for each variable (Ncars, Fuel_type and Euro_norm).
The income effect was also tested using the script `src/validation/income_effect.py`.

The trained classifiers for all predicted variables are provided in the `trained_model_Paris_EGT2020/` folder.

(*) EGT H2020-Île-de-France Mobilités-OMNIL-DRIEAT, see [survey description online](https://omnil.fr/egt-h2020)
