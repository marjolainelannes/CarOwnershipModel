##################################################################################
# Purpose: Preprocess data
# Note: outputs are the training sets (Nvehicles and Vehicle_types), a synthesis of all informations at the household_id level
#       (Population) and statistics on fuel types
##################################################################################
import os.path
import pandas as pd
import sys
sys.path.append("/include/")
from numeric import countX, get_indexes
from new_municipalities import new_municipalities
from vehicle_type import emission_standard
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input / output files
data_dir = "../../data/"
households_f = data_dir + "EGT20/01_Menage_egt1820.xlsx"
individuals_f = data_dir + "cleaned_individuals.csv"
cars_f = data_dir + "/EGT20/05_Voiture_egt1820.xlsx"
accessibility_f = data_dir + "Accessibility-score.csv"
output_dir = "../../training_datasets/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
f_output_pop = output_dir + "Population.csv"
f_output_nveh = output_dir + "Nvehicles.csv"
f_output_vehtype = output_dir + "Vehicle_types.csv"

# Load data
Household = pd.read_excel(households_f)
Individuals = pd.read_csv(individuals_f)
Car = pd.read_excel(cars_f)
Accessibility = pd.read_csv(accessibility_f)
Car_owners = list(Car.IDCEREMA)  # list of households owning cars (with duplicates)
individuals_households = list(Individuals.IDCEREMA) # list of individuals's households

# Analyze data
Household['POIDSM'] = pd.to_numeric(Household['POIDSM'].str.replace(',', '.'))
pop_weight = sum(Household.POIDSM)
print(Household.shape[0], 'households before cleaning')

# Input data setup
Individuals['DDOMTRAV'] = pd.to_numeric(Individuals['DDOMTRAV'].str.replace(',', '.'))
for i, hh in enumerate(Household['IDCEREMA']):
    if pd.isnull(Household.loc[i,'NB_VD']) :
        Household.loc[i,'NB_VD']=0
Car.ENERG = Car["ENERG"].astype(str)
Car['ENERG2'] = Car['ENERG'].map({"1":'Diesel',"2":'Petrol',"3":'Hybrid', "4":'Diesel', "5":"Hybrid", "6":"Electric",
                                  "7":"Petrol","-1":"others","9":"others"})
for car, energy in enumerate(Car['ENERG2']) :
    if energy == "others" :
        detail = Car.loc[car, "MDL"]
        if (detail in ['nsp', 'ne sait pas',None]) or (detail =="Clio" and Car.loc[car, "ENERG_txt"]=="NSP"):
            Car.loc[car, 'ENERG2'] = np.nan
            id_household = Car.loc[car, 'IDCEREMA']
            Household.drop(Household[Household.IDCEREMA == id_household].index, inplace=True)
        else :
            Car.loc[car, 'ENERG2'] = "Petrol"
Car.drop(Car[Car.ENERG2 == np.nan].index, inplace=True)
Household.reset_index(drop=True, inplace=True)

# Statistics no answer
no_income = Household[Household.REVENU < 0]
no_housing_type = Household[Household.TYPELOG == -1]
print(no_income.shape[0], "respondents did not answer for income, accounting for ", sum(list(no_income['POIDSM'])),
      "households:", sum(list(no_income['POIDSM']))*100/pop_weight,"% pop")
print(no_housing_type.shape[0], "respondents did not answer for housing type, accounting for ", sum(list(no_housing_type['POIDSM'])),
      "households:", sum(list(no_housing_type['POIDSM']))*100/pop_weight,"% pop")

# clean data
Household['Mean_income'] = Household['REVENU'].map({-2:np.nan,-1:np.nan, 1:600, 2:1000, 3:1400, 4:1800, 5:2200, 6:2700, 7:3250, 8:4000, 9:5000, 10:7000})
Household["TYPELOG"] = Household["TYPELOG"].map({1:"house",2:"flat",9:"others", -1:None})
Household["TYPE_MEN"] = Household["TYPE_MEN"].map({'Femme seule':'SW','Couple sans enfant':'CWOC','Homme seul':'SM','Famille monoparentale mère':'SPFM', 'Couple avec enfant':'CWC', 'Famille monoparentale père':'SPFF', 'Autre':'others'})
n_voi = Car.shape[0]
n_ind = Individuals.shape[0]
n_hh = Household.shape[0]
print(n_hh, 'households after cleaning, then ', 1-sum(Household.POIDSM)/pop_weight, " share of the households where removed")

# Initialize dataframes
Variables = ["household_id", "age", "income", "N_workers", "household_type", "PT_share_work", "parking",
             "commuting_distance", "parking_at_workplace", "housing_type", "PT_share_home","weight",
             "N_cars", "Vehicle_types", "Fuel_type_1","Euro_norm_1","Fuel_type_2", "Euro_norm_2", "Fuel_type_3", "Euro_norm_3",
             "Fuel_type_4", "Euro_norm_4","Fuel_type_5", "Euro_norm_5", "Fuel_type_6", "Euro_norm_6"]
Input = pd.DataFrame(columns=Variables, index=range(0, n_hh))
Types = pd.DataFrame(columns=["Car_type", "N_cars"])

cnt_cars = 0
for i in range (0, n_hh) :
    ### MUNICIPALITY ID ####
    municipality = int(Household.loc[i, "RESCOMM"])
    municipality = int(new_municipalities(municipality))
    ### HOUSEHOLD CHARACTERISTICS ###
    household_id = str(Household.loc[i, 'IDCEREMA'])
    household_members = get_indexes(individuals_households, household_id)
    age_max = 0
    # local and regional variables
    index_municipality = list(Accessibility.M_code).index(municipality)
    pt_share_home = Accessibility.loc[index_municipality, "PT_share"]
    Input.loc[i, "PT_share_home"] = pt_share_home
    commuting_dist = 0
    parking_at_workplace = 0
    # Define the longest commuting distance and the working city associated to it + maximal age
    for person in household_members :
        if Individuals.loc[person, 'PKVPTRAV'] == 1 :
            parking_at_workplace = 1
        age_max = np.nanmax([age_max, Individuals.loc[person, "TRAGE"]])
        commuting_dist_pers = Individuals.loc[person, "DDOMTRAV"]
        commuting_dist = np.nanmax([commuting_dist, commuting_dist_pers])
        if commuting_dist == commuting_dist_pers :
            pt_share_work = Individuals.loc[person, "PT_SHARE_WORK"]
    # PT_share_work
    if not np.isnan(commuting_dist):
        Input.loc[i, "PT_share_work"] = pt_share_work
    else:
        Input.loc[i, "PT_share_work"] = None
    # Socioeconomic characteristics
    Input.loc[i, "household_id"] = household_id
    Input.loc[i, "age"] = age_max
    weight = Household.loc[i, 'POIDSM']
    Input.loc[i, "weight"] = weight
    HH_cars_indexes = get_indexes(Car_owners, household_id)
    ## Income level
    if Household.loc[i, "REVENU"] > 0 :
        Input.loc[i, "income"] = np.log(Household.loc[i, "Mean_income"]) #np.log
    else :
        Input.loc[i, "income"] = None
    ## Number of active workers
    n_workers = Household.loc[i, "MNPACT"]
    if n_workers < 2 :
        Input.loc[i, "N_workers"] = str(n_workers)
    else :
        Input.loc[i, "N_workers"] = "2+"
    ## Household type (categories translated later)
    Input.loc[i, "household_type"] = Household.loc[i, "TYPE_MEN"]
    ## Housing type
    Input.loc[i, "housing_type"] = Household.loc[i, "TYPELOG"]
    ### MOBILITY VARIABLES ####################
    ## Parking at workingplace
    Input.loc[i, "parking_at_workplace"] = parking_at_workplace
    ## Commuting distance
    if commuting_dist < 0 :
        Input.loc[i, "commuting_distance"] = None
    else :
        Input.loc[i,"commuting_distance"] = commuting_dist
    # Number of cars
    n_cars_hh = int(Household.loc[i, "NB_VD"])
    Input.loc[i, "N_cars"] = n_cars_hh
    cnt_cars += n_cars_hh * weight
    construction_year = 0
    parking = 0
    # Cars types (fuel type and euro norm)
    if n_cars_hh > 0 :
        hh_cars = [""]*n_cars_hh
        vehicles = str()
        for car in HH_cars_indexes :
            n = Car.loc[car, "NVP"]
            # Emission standard
            construction_year = Car.loc[car, "APMC"]
            EN_name = "Euro_norm_" + str(n)
            Input.loc[i, EN_name] = str(emission_standard(int(construction_year)))
            # Fuel type
            fuel_type = Car.loc[car, "ENERG2"]
            FT_name = "Fuel_type_" + str(n)
            Input.loc[i, FT_name] = fuel_type
            hh_cars[n - 1] = fuel_type
            # presence of parking at home
            if str(Car.loc[car, "STAT"]) == "1" :
                parking = 1
        # Synthesis of all car fuel type
        for j in range(0, n_cars_hh) :
            fuel_type = hh_cars[j]
            if j == 0 :
                vehicles = vehicles + fuel_type
            elif j in [1, 2, 3]:
                vehicles = vehicles + " - " + fuel_type
        Input.loc[i, "Vehicle_types"] = vehicles
        vehicle_types = list(Types.Car_type)
        # Statistics on car fuel types pairs
        if vehicles not in vehicle_types :
            new = [[vehicles, n_cars_hh]]
            new_row = pd.DataFrame(new, columns = ["Car_type","N_cars"])
            Types = pd.concat([Types, new_row], ignore_index=True)
    Input.loc[i, "parking"] = parking
Input['age'] = Input['age'].map({0:'0 to 5', 5:'5 to 14', 15:'15 to 24', 25:'25 to 34', 35:'35 to 54', 55:'55 to 64',
                                65:'65 to 74', 75:'75+'})
print("Total car fleet :",cnt_cars,"cars")

##### CREATE TWO DATASETS FOR STAGE 1 (NUMBER OF CARS) AND STAGE 2 (CAR TYPE) #########

### CREATE N_CARS DATASET ###
Input_nveh = Input.drop(columns=['Vehicle_types'],inplace=False)
Input_nveh = Input_nveh.astype({'N_cars': 'str'})
Input_nveh.drop(Input_nveh[Input_nveh.N_cars == None].index, inplace=True)
Input_nveh['N_cars'] = Input_nveh['N_cars'].map({'0':'0','1':'1','2':'2+','3':'2+','4':'2+','5':'2+','6':'2+'})

##### CREATE CAR_TYPE DATASET ###################
Variables_car_type = Variables + ['Fuel_type', 'Euro_norm']
Input_Car_type = pd.DataFrame(columns=Variables_car_type)
n_hh = Input.shape[0]
index_list = Input.index
for i in index_list :
    n_veh_men = int(Input.loc[i, "N_cars"])
    if n_veh_men > 0 :
        for j in range(1,n_veh_men+1) :
            new_row = pd.DataFrame(columns=Variables_car_type)
            # print(Input[Input.index == i], Input[Input.index == i].shape)
            new_row[Variables] = Input[Input.index == i] #plus deux nan
            CY_name = "Euro_norm_" + str(j)
            new_row['Euro_norm'] = Input.loc[i, CY_name]
            FT_name = "Fuel_type_" + str(j)
            new_row['Fuel_type'] = Input.loc[i,FT_name]
            Input_Car_type = pd.concat([Input_Car_type, new_row], ignore_index=True)
Input_Car_type = Input_Car_type.astype({'N_cars': 'str'})
Input_Car_type.drop(Input_Car_type[Input_Car_type.N_cars == None].index, inplace=True)
Input_Car_type['N_cars'] = Input_Car_type['N_cars'].map({'0': '0', '1': '1', '2': '2+', '3': '2+', '4': '2+', '5': '2+', '6': '2+'})
Input_Car_type.drop(columns=['Vehicle_types'], inplace=True)
for i in range(1, 7):
    fuel = "Euro_norm_" + str(i)
    year = "Fuel_type_" + str(i)
    Input_nveh.drop(columns=[fuel, year], inplace=True)
    Input_Car_type.drop(columns=[fuel, year], inplace=True)

##### SUMMARY STATISTICS ON CAR TYPES COMBINAISONS #####################

n_vehicle_types = Types.shape[0]
Fuel_types = pd.DataFrame(columns=['Combination of vehicles', 'N_cars','Number of occurences'], index=range(0, n_vehicle_types))
Vehicles_EGT = list(Input.Vehicle_types)
for i in range (0, n_vehicle_types) :
    type = Types.loc[i, 'Car_type']
    Fuel_types.loc[i,'Combination of vehicles'] = type
    Fuel_types.loc[i,'Number of occurences'] = countX(Vehicles_EGT, type)
    Fuel_types.loc[i, 'N_cars'] = Types.loc[i, 'N_cars']
    print('{} has occurred {} times'.format(type, countX(Vehicles_EGT, type)))

# stat Ncars
Input = Input.astype({'N_cars': 'str'})
Input.drop(Input[Input.N_cars == None].index, inplace=True)
Input['N_cars'] = Input['N_cars'].map({'0': '0', '1': '1', '2': '2+', '3': '2+', '4': '2+', '5': '2+', '6': '2+'})
fig, ax = plt.subplots()
ax = sns.histplot(data=Input, x="N_cars", weights="weight", color="cornflowerblue")
ax.set(xlabel = "Number of cars owned", ylabel = "Number of households", title = "Distribution of cars owned by households")
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(p.get_height()*100/s),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
fname = output_dir + "Ncars.png"
plt.savefig(fname, format='png', dpi=300)
plt.show()

# stat Fuel type
fig, ax = plt.subplots()
ax = sns.histplot(data=Input_Car_type, x="Fuel_type", weights="weight",color="cornflowerblue")
ax.set(xlabel = "Fuel type", ylabel = "Number of cars", title = "Distribution of car fuel types")
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(p.get_height()*100/s),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
fname = output_dir + "Fuel_type.png"
plt.savefig(fname, format='png', dpi=300)
plt.show()

# stat Euro_norm
fig, ax = plt.subplots()
ax = sns.histplot(data=Input_Car_type, x="Euro_norm", weights="weight", color="cornflowerblue")
ax.set(xlabel = "Emission standard", ylabel = "Number of cars", title = "Distribution of car emission standards")
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(p.get_height()*100/s),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
fname = output_dir + "Euro_norm.png"
plt.savefig(fname, format='png', dpi=300)
plt.show()

# Save dataframes
Input.drop(columns=['weight'],inplace=True)
Input.to_csv(f_output_pop, index = False)
Input_nveh.drop(columns=['weight'],inplace=True)
Input_nveh.to_csv(f_output_nveh, index=False)
Input_Car_type.drop(columns=['weight'],inplace=True)
Input_Car_type.to_csv(f_output_vehtype, index=False)
