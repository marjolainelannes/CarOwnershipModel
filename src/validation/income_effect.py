##################################################################################
# Purpose: Check if the model reproduces an income effect on car ownership
# Notes: Apply the model to the EGT dataset
##################################################################################

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

# Input / output files
input_dir = "../../training_dataset/"
income_effect_stat_EGT = "../../output/income_effect/"
output_dir = income_effect_stat_EGT + "model/"
car_type_EGT_f = input_dir + "Vehicle_types.csv"
households_EGT_f = input_dir + "Nvehicles.csv"
household_f = "../../data/EGT20/01_Menage_egt1820.xlsx"
pipeline_dir = "../../trained_model_Paris_EGT2020/"
pipeline_Ncars = pipeline_dir + 'pipeline_N_cars'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and clean data
car_type_EGT = pd.read_csv(car_type_EGT_f)
car_type_EGT = car_type_EGT.astype({'household_id':str})
households_EGT = pd.read_csv(households_EGT_f)
print(car_type_EGT.columns)
Household = pd.read_excel(household_f)
Household = Household.astype({'IDCEREMA':str})
Household['POIDSM'] = Household['POIDSM'].str.replace(',', '.').astype(float)

# Assign weight to car types and households
for car_idx, household_id in enumerate(car_type_EGT["household_id"]) :
    household_index = Household[Household.IDCEREMA == household_id].index[0]
    car_type_EGT.loc[car_idx, "weight"] = Household.loc[household_index, "POIDSM"]
for household_idx, household_id in enumerate(households_EGT["household_id"]) :
    household_index = Household[Household.IDCEREMA == household_id].index[0]
    households_EGT.loc[household_idx, "weight"] = Household.loc[household_index, "POIDSM"]

# Calculate the share of households motorization classes
n_households = households_EGT['weight'].sum()
print("Number of households", n_households)
income_bins = [1, 800, 1200, 1600, 2000, 2400, 3000, 3500, 4500, 5500, 100000]
income_bin_labels = [("< "+str(income_bins[i+1]))+"€" for i in range(len(income_bins)-1)]
income_bin_labels[-1] = '> 5500€'
print(income_bin_labels)
income_bins_log = [np.log(x) for x in income_bins]
motorization_classes = ['0','1','2+']
population_composition = pd.DataFrame(columns=["N_cars", "Income", "Share"])
population_by_income = pd.DataFrame(columns=["Income", '0','1','2+'])
for i in range(len(income_bins)-1):
    extract_1 = households_EGT[(households_EGT.income > income_bins_log[i]) & (households_EGT.income <= income_bins_log[i + 1])]
    n_income_range = extract_1['weight'].sum()
    new_row_income = pd.DataFrame(data=[income_bin_labels[i]],columns=["Income"])
    for motorization in motorization_classes :
        extract_2 = extract_1[extract_1.N_cars == motorization]
        n_subsegment = extract_2['weight'].sum()
        new_row = pd.DataFrame(columns=["N_cars", "Income", "Share"])
        new_row.loc[0, "Income"] = income_bin_labels[i]
        new_row.loc[0, "N_cars"] = motorization
        new_row.loc[0, "Share"] = 100*n_subsegment / n_households
        population_composition = pd.concat([population_composition, new_row],ignore_index=True)
        new_row_income.loc[0, motorization] = 100 * n_subsegment / n_income_range
    population_by_income = pd.concat([population_by_income, new_row_income], ignore_index=True)
print(population_by_income)
population_by_income.to_csv(income_effect_stat_EGT+"income_effect_on_motorization_EGT.csv", index=False)
print("Stats EGT fuel_type, sum of shares:",sum(population_composition['Share']))

# Generates and plot statistics on the households motorization within the EGT population
fig, ax = plt.subplots()
ax = sns.barplot(data=population_composition, x="Income", y="Share", hue="N_cars")
ax.set(xlabel = "Household income", ylabel = "Share of households (%)",
       title = "Distribution of motorization categories (EGT 2018-2020)")
plt.legend(loc='upper left')
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_motorization_distribution_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Generates and plot statistics on the household motorization by income range within the EGT population
fig, ax = plt.subplots()
population_by_income.set_index("Income").plot(kind='bar', stacked=True)
plt.xlabel("") # plt.xlabel("Household income (€)")
plt.ylabel("Share of population within income class (%)")
plt.title("Distribution of motorization categories by income range (EGT 2018-2020)")
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
          fancybox=True, shadow=True, ncol=5)
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_motorization_by_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Calculate the share of each car type
n_cars = car_type_EGT['weight'].sum()
print("Car fleet size", n_cars)
income_bins = [1, 800, 1200, 1600, 2000, 2400, 3000, 3500, 4500, 5500, 100000]
income_bin_labels = [("< "+str(income_bins[i+1]))+"€" for i in range(len(income_bins)-1)]
income_bin_labels[-1] = '> 5500€'
print(income_bin_labels)
income_bins_log = [np.log(x) for x in income_bins]
fuel_types = ['Diesel','Petrol','Electric','Hybrid']
car_fleet_composition = pd.DataFrame(columns=["Fuel_type", "Income", "Share"])
car_fleet_by_income = pd.DataFrame(columns=["Income", 'Diesel','Petrol','Electric','Hybrid'])
for i in range(len(income_bins)-1):
    extract_1 = car_type_EGT[(car_type_EGT.income > income_bins_log[i]) & (car_type_EGT.income <= income_bins_log[i + 1])]
    n_income_range = extract_1['weight'].sum()
    new_row_income = pd.DataFrame(data=[income_bin_labels[i]],columns=["Income"])
    for fuel_type in fuel_types :
        extract_2 = extract_1[extract_1.Fuel_type == fuel_type]
        n_subsegment = extract_2['weight'].sum()
        new_row = pd.DataFrame(columns=["Fuel_type", "Income", "Share"])
        new_row.loc[0, "Income"] = income_bin_labels[i]
        new_row.loc[0, "Fuel_type"] = fuel_type
        new_row.loc[0, "Share"] = 100*n_subsegment / n_cars
        car_fleet_composition = pd.concat([car_fleet_composition, new_row],ignore_index=True)
        new_row_income.loc[0, fuel_type] = 100 * n_subsegment / n_income_range
    car_fleet_by_income = pd.concat([car_fleet_by_income, new_row_income], ignore_index=True)
print(car_fleet_by_income)
car_fleet_by_income.to_csv(income_effect_stat_EGT+"income_effect_on_fuel_type_EGT.csv", index=False)
print("Stats EGT fuel_type, sum of shares:",sum(car_fleet_composition['Share']))

# Generates and plot statistics on the car types within the EGT car fleet
fig, ax = plt.subplots()
ax = sns.barplot(data=car_fleet_composition, x="Income", y="Share", hue="Fuel_type")
ax.set(xlabel = "Household income", ylabel = "Share of total car fleet (%)",
       title = "Distribution of fuel types (EGT 2018-2020)")
plt.legend(loc='upper left')
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_fuel_types_distribution_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Generates and plot statistics on the car types by income range within the EGT car fleet
fig, ax = plt.subplots()
car_fleet_by_income.set_index("Income").plot(kind='bar', stacked=True)
plt.xlabel("") # plt.xlabel("Household income (€)")
plt.ylabel("Share of car fleet within income class (%)")
plt.title("Distribution of fuel types by income range (EGT 2018-2020)")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
          fancybox=True, shadow=True, ncol=5)
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_fuel_type_by_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Statistics euro norm
euro_norms = ['<ECE', 'Euro-1', 'Euro-2', 'Euro-3','Euro-4','Euro-5','Euro-6']
car_fleet_composition = pd.DataFrame(columns=["Income", "Euro_norm", "Share"])
car_fleet_by_income = pd.DataFrame(columns=["Income",'<ECE', 'Euro-1', 'Euro-2', 'Euro-3','Euro-4','Euro-5','Euro-6'])
for i in range(len(income_bins)-1):
    extract_1 = car_type_EGT[(car_type_EGT.income > income_bins_log[i]) & (car_type_EGT.income <= income_bins_log[i+1])]
    n_income_range = extract_1['weight'].sum()
    new_row_income = pd.DataFrame(data=[income_bin_labels[i]], columns=["Income"])
    for euro_norm in euro_norms :
        extract_2 = extract_1[extract_1.Euro_norm == euro_norm]
        n_subsegment = extract_2['weight'].sum()
        new_row = pd.DataFrame(columns=["Income", "Euro_norm", "Share"])
        new_row.loc[0, "Income"] = income_bin_labels[i]
        new_row.loc[0, "Euro_norm"] = euro_norm
        new_row.loc[0, "Share"] = 100*n_subsegment / n_cars
        car_fleet_composition = pd.concat([car_fleet_composition, new_row],ignore_index=True)
        new_row_income.loc[0, euro_norm] = 100 * n_subsegment / n_income_range
    car_fleet_by_income = pd.concat([car_fleet_by_income, new_row_income], ignore_index=True)
print(car_fleet_by_income)
print("Stats EGT fuel_type, sum of shares:",sum(car_fleet_composition['Share']))
car_fleet_by_income.to_csv(income_effect_stat_EGT+"income_effect_on_euro_norm_EGT.csv", index=False)

# Generates and plot statistics on the car types within the synthetic car fleet
fig, ax = plt.subplots()
ax = sns.barplot(data=car_fleet_composition, x="Income", y="Share", hue="Euro_norm")
ax.set(xlabel = "Household income", ylabel = "Share of total car fleet (%)",
       title = "Distribution of emission standards (EGT 2018-2020)")
plt.legend(loc='upper left')
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_euro_norm_distribution_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Generates and plot statistics on the car types by income range within the EGT car fleet
fig, ax = plt.subplots()
car_fleet_by_income.set_index("Income").plot(kind='bar', stacked=True)
plt.xlabel("") # plt.xlabel("Household income (€)")
plt.ylabel("Share of car fleet within income class (%)")
plt.title("Distribution of emission standards by income range (EGT 2018-2020)")
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
          fancybox=True, shadow=True, ncol=7)
plt.xticks(rotation=15)
fig_file = income_effect_stat_EGT + "EGT_euro_norm_by_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

#################################################################################################

# apply Ncars classifier
test_data_Ncars = households_EGT.copy()
test_data_Ncars.drop(columns=['household_id',"weight"], inplace=True)
households_infos = households_EGT[["household_id","weight"]]
print(test_data_Ncars.columns)
Ncars_model = load_model(pipeline_Ncars)
prediction_Ncars = predict_model(Ncars_model, raw_score = True, data=test_data_Ncars)
prediction_Ncars = pd.concat([households_infos, prediction_Ncars], axis=1).reset_index(drop=True)
Ncars_output_f = output_dir + "Ncars_model_prediction.csv"
prediction_Ncars.to_csv(Ncars_output_f, index=False)

# Income effect Ncars
n_households = prediction_Ncars['weight'].sum()
car_fleet_composition = pd.DataFrame(columns=["Income", 'N_cars', "Share"])
population_by_income = pd.DataFrame(columns=["Income", '0','1','2+'])
for i in range(len(income_bins) - 1):
    extract = prediction_Ncars[
        (prediction_Ncars.income > income_bins_log[i]) & (prediction_Ncars.income <= income_bins_log[i + 1])]
    n_income_range = extract['weight'].sum()
    new_row_income = pd.DataFrame(data=[income_bin_labels[i]], columns=["Income"])
    total_share = 0
    for var_j in ['0', '1', '2+']:
        output_var = "prediction_score_" + var_j
        share_subsegment = np.average(extract[output_var], weights=extract['weight'])
        new_row = pd.DataFrame(columns=["Income", 'N_cars', "Share"])
        new_row.loc[0, "Income"] = income_bin_labels[i]
        new_row.loc[0, 'N_cars'] = var_j
        new_row.loc[0, "Share"] = 100 * share_subsegment * n_income_range / n_households
        car_fleet_composition = pd.concat([car_fleet_composition, new_row], ignore_index=True)
        new_row_income.loc[0, var_j] = 100 * share_subsegment
        total_share += share_subsegment
    for var_j in ['0', '1', '2+']:
        new_row_income.loc[0, var_j] = new_row_income.loc[0, var_j] / total_share
    population_by_income = pd.concat([population_by_income, new_row_income], ignore_index=True)

# Generates and plot statistics on the household motorization within the synthetic population
fig, ax = plt.subplots()
ax = sns.barplot(data=car_fleet_composition, x="Income", y="Share", hue="N_cars")
ax.set(xlabel="Household income", ylabel="Share of total car fleet (%)", title="Distribution of motorization categories (modeled)")
plt.legend(loc='upper left')
plt.xticks(rotation=15)
fig_file = output_dir + "modeled_motorization_distribution_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Generates and plot statistics on the household motorization by income range within the EGT population
fig, ax = plt.subplots()
population_by_income.set_index("Income").plot(kind='bar', stacked=True)
plt.xlabel("") # plt.xlabel("Household income (€)")
plt.ylabel("Share of population within income class (%)")
plt.title("Distribution of motorization categories by income range (EGT 2018-2020)")
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])
# Put a legend below current axis
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), # loc='upper center', bbox_to_anchor=(0.5, -0.05)
          fancybox=True, shadow=True, ncol=5)
plt.xticks(rotation=15)
fig_file = output_dir + "modeled_motorization_by_income.png"
plt.savefig(fig_file, format='png', dpi=1000)

# Ncars_random_draw
for i in range(households_EGT.shape[0]) :
    probabilities = np.array([prediction_Ncars.loc[i, 'prediction_score_0'], prediction_Ncars.loc[i, 'prediction_score_1'],
                             prediction_Ncars.loc[i, 'prediction_score_2+']])
    probabilities /= probabilities.sum()  # normalize
    prediction_Ncars.loc[i, 'N_cars'] = np.random.choice(['0', '1', '2+'], p=list(probabilities))
households_ncars = prediction_Ncars.drop(columns=['prediction_label', 'prediction_score_0','prediction_score_1','prediction_score_2+'])
def generate_number_of_cars(x):
    if x == "0":
        return 0
    elif x == "1":
        return 1
    elif x == "2+":
        p = np.random.choice(np.arange(2, 7), p=[0.86588, 0.11735, 0.01425, 0.00168, 0.00084])
        return(p)
households_ncars["Ncars_updated"] = households_ncars["N_cars"].apply(generate_number_of_cars)
Ncars_output_f = output_dir + "Ncars_model_prediction_updated.csv"
households_ncars.to_csv(Ncars_output_f, index=False)

# generate cars database
car_columns = list(households_ncars.columns) +['Car_number']
households_with_cars = pd.DataFrame(columns=car_columns)
for i in range(households_EGT.shape[0]):
    n_cars = households_ncars.loc[i, "Ncars_updated"]
    if n_cars > 0:
        new_row = households_ncars[households_ncars.index == i].reset_index(drop=True)
        for j in range(0, n_cars):
            new_row.loc[0, "Car_number"] = j + 1
            #print(new_row.columns, households_with_cars.columns)
            households_with_cars = pd.concat([households_with_cars, new_row], ignore_index=True)
print(households_with_cars)

# apply fuel type classifier
variables = ["Fuel_type","Euro_norm"] # "Fuel_type",
indexes = [variables.index(item) for item in variables]
n_cars = households_with_cars['weight'].sum()#.iloc[0]
print(n_cars, "cars in the generated database")
households_ids = households_with_cars[["household_id", "weight", "Car_number"]]
households_with_cars.drop(columns=["household_id", "weight","Ncars_updated", "Car_number"], axis=1, inplace=True)
print("input_car_types:",households_with_cars.columns)
for i in indexes :
    var = variables[i]
    print(var)
    ## Model prediction ##
    print("Model prediction...")
    name = pipeline_dir + 'pipeline_' + var # + '.pkl'
    model = load_model(name)
    # apply the model to the synthetic population
    prediction = predict_model(model, data=households_with_cars, raw_score=True)
    households_with_cars_updated = pd.concat([households_ids, prediction], axis=1).reset_index(drop=True)
    print(households_with_cars_updated.columns)
    # save output
    output_f = output_dir + var + "_model_prediction.csv"
    prediction.to_csv(output_f, index=False)

    # stats
    if var == "Euro_norm" :
        var_labels = ['<ECE', 'Euro-1', 'Euro-2', 'Euro-3', 'Euro-4', 'Euro-5', 'Euro-6']
    else :
        var_labels = ['Diesel','Petrol','Electric','Hybrid']
    car_fleet_composition = pd.DataFrame(columns=["Income", var, "Share"])
    car_fleet_by_income = pd.DataFrame(columns=["Income"]+var_labels)
    for i in range(len(income_bins) - 1):
        extract = households_with_cars_updated[(households_with_cars_updated.income > income_bins_log[i]) & (
                    households_with_cars_updated.income <= income_bins_log[i + 1])]
        n_income_range = extract['weight'].sum()
        new_row_income = pd.DataFrame(data=[income_bin_labels[i]], columns=["Income"])
        total_share = 0
        for var_j in var_labels:
            output_var = "prediction_score_" + var_j
            share_subsegment = np.average(extract[output_var], weights=extract['weight'])
            new_row = pd.DataFrame(columns=["Income", var, "Share"])
            new_row.loc[0, "Income"] = income_bin_labels[i]
            new_row.loc[0, var] = var_j
            new_row.loc[0, "Share"] = 100 * share_subsegment * n_income_range / n_cars
            car_fleet_composition = pd.concat([car_fleet_composition, new_row], ignore_index=True)
            new_row_income.loc[0, var_j] = 100 * share_subsegment
            total_share += share_subsegment
        for var_j in var_labels:
            new_row_income.loc[0, var_j] = new_row_income.loc[0, var_j] / total_share
        car_fleet_by_income = pd.concat([car_fleet_by_income, new_row_income], ignore_index=True)
    print(car_fleet_composition.head())
    print("Modeled ", var, "sum of shares:", sum(car_fleet_composition['Share']))
    car_fleet_composition = car_fleet_composition[car_fleet_composition.Share > 0]

    # Generates and plot statistics on the car types within the synthetic car fleet
    if var == "Fuel_type":
        name = "fuel types"
    else :
        name = "emission standards"
    fig, ax = plt.subplots()
    ax = sns.barplot(data=car_fleet_composition, x="Income", y="Share", hue=var)
    ax.set(xlabel="Household income", ylabel="Share of total car fleet (%)", title="Distribution of car "+name + " (modeled)")
    plt.legend(loc='upper left')
    plt.xticks(rotation=15)
    fig_file = output_dir + "modeled_" + var + "_distribution_income.png"
    plt.savefig(fig_file, format='png', dpi=1000)

    # Generates and plot statistics on the car types by income range within the EGT car fleet
    fig, ax = plt.subplots()
    car_fleet_by_income.set_index("Income").plot(kind='bar', stacked=True)
    plt.xlabel("")  # plt.xlabel("Household income (€)")
    plt.ylabel("Share of car fleet within income class (%)")
    plt.title("Distribution of car "+ name + " by income range")
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])
    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
               fancybox=True, shadow=True, ncol=7)
    plt.xticks(rotation=15)
    fig_file = output_dir + "modeled_" + var + "_by_income.png"
    plt.savefig(fig_file, format='png', dpi=1000)

