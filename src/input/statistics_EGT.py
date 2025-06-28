##################################################################################
# Purpose: Generate statistics of EGT 2018-2020 mobility survey in Paris region
##################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append("/include/")
from numeric import decimal_converter

##### LOAD DATA #################################################################

# Input / output files
data_dir = "../../data/EGT20/"
households_f = data_dir +  "EGT20/01_Menage_egt1820.xlsx"
individuals_f = data_dir + "EGT20/02_Individu_egt1820.xlsx"
cars_f = data_dir + "EGT20/05_Voiture_egt1820.xlsx"
accessibility_f = data_dir + "Accessibility-score.csv"
statistics_dir = data_dir + "EGT20_statistics/"
if not os.path.exists(statistics_dir):
    os.makedirs(statistics_dir)

# Load data
Household = pd.read_excel(households_f, converters={'POIDSM':decimal_converter})
Ind = pd.read_excel(individuals_f, converters={'POIDSI':decimal_converter, 'DDOMTRAV':decimal_converter})
Car = pd.read_excel(cars_f)
Access = pd.read_csv(accessibility_f)

##### PRINT STATISTICS: DATABASE STRUCTURE ######################################

## Preprocessing households and individuals datasets ##

Ind = Ind.merge(Household[['IDCEREMA','POIDSM']], on='IDCEREMA', how='left')
Household['POIDSM'] = Household['POIDSM'].astype(float)
Household['TYPE_MEN'] = Household['TYPE_MEN'].map({'Femme seule':'Single woman','Couple sans enfant':'Couple w/o child',
                                                   'Homme seul':'Single man','Famille monoparentale mère':'MF mother',
                                                   'Couple avec enfant':'Couple w/ child', 'Famille monoparentale père':'MF father',
                                                   'Autre':'Others'})
for i, hh in enumerate(Household['IDCEREMA']):
    if pd.isnull(Household.loc[i,'NB_VD']) :
        Household.loc[i,'NB_VD']=0
Household['REVENU'] = Household['REVENU'].map({int(-2):None, int(-1):None, 1:700, 2:800, 3:1200, 4:1600, 5:2000, 6:2400, 7:3000, 8:3500, 9:4500, 10:5500})
for i in range(Household.shape[0]):
    Household.loc[i,'OTHER_VEH'] = Household.loc[i,'NB_VEH'] - Household.loc[i,'NB_VD']

## Database statistics ##
n_ind = Ind.shape[0]
n_households = Household.shape[0]
n_cars = Car.shape[0]
index_list = Car.index
print("Number of individuals: ", n_ind)
print("Number of households: ", n_households)
print("Number of cars: ", n_cars)

## Preprocess household equipments dataset ##
Car = Car.merge(Household[['IDCEREMA','NB_VD','POIDSM']], on='IDCEREMA', how='left')
Car.ENERG = Car["ENERG"].astype(str)
Car = Car.rename(columns = {"NVP":"Car number", "NB_VD":"Number of cars owned"})
for i, year in enumerate(Car['APMC']) :
    if (not np.isnan(year)) and (int(year)<1993):
        Car.loc[i,'APMC'] = 1990
Car['ENERG2'] = Car['ENERG'].map({"1":'Diesel',"2":'Petrol',"3":'Petrol hybrid', "4":'Diesel',
                                 "5":"Petrol hybrid", "6":"Electric", "7":"Petrol","9":None, "-1":None})
Car['ENERG'] = Car['ENERG'].map({"1":'Diesel',"2":'Petrol',"3":'Petrol hybrid', "4":'Diesel hybrid',
                                 "5":"Plug-in hybrid", "6":"Electric", "7":"LPG","9":None, "-1":None})
Household.rename(columns = {"NB_VD":"Number of cars owned"}, inplace=True)
output_dir = path + "temp_EGT20/training_dataset/"
f_output_nveh = output_dir + "Nvehicles.csv"
Training_population = pd.read_csv(f_output_nveh)
Training_population = Training_population.merge(Household[['IDCEREMA','POIDSM']], left_on='household_id',
                                                right_on='IDCEREMA', how='left')


##### PART 1. STATISTICS ON MULTI-MOTORIZED HOUSEHOLDS ##################################

## Fuel Type Histogram
fig, ax = plt.subplots()
ax = sns.histplot(data=Car, x="ENERG", weights="POIDSM", binwidth = 1 ,discrete='True', color="cornflowerblue")
ax.set(xlabel='Fuel type', ylabel='Number of cars', title='Distribution of fuel types \n (EGT 2018-2020)')
plt.xticks(rotation=18)
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
fname = statistics_dir + "Fuel_type.png"
plt.savefig(fname, format='png', dpi=400, bbox_inches='tight')
plt.show()

##### PART 2. STATISTICS ON HOUSEHOLDS CHARACTERISTICS ##################################

### Households weights distribution ###
fig, ax = plt.subplots()
weight = pd.to_numeric(Household.loc[:,'POIDSM'])
ax = sns.kdeplot(x=np.array(weight), shade=True, color="cornflowerblue")
ax.set(xlabel='Household weight', ylabel='Density', title='Density plot of households weights \n (EGT 2018-2020)')
fname = statistics_dir + "weight.png"
plt.savefig(fname, format='png', dpi=400, bbox_inches='tight')
plt.show()

## Histogram of the age of respondents
fig, ax = plt.subplots()
age_bins = [0,5,15,25,35,55,65,75,100]
ax = sns.histplot(data=Ind, x="TRAGE", weights="POIDSM", bins=age_bins, color="cornflowerblue")
ax.set(xlabel='Age', ylabel='Number of individuals', title='Distribution of the respondents age \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            '{}%'.format(int(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(ticks=age_bins)
fname = statistics_dir + "age_individuals.png"
plt.savefig(fname, format='png', dpi=400, bbox_inches='tight')
plt.show()

## Histogram of households income level
income_bins = [100, 800, 1200, 1600, 2000, 2400, 3000, 3500, 4500, 5500, 7000]
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="REVENU", weights="POIDSM", bins=income_bins, color="cornflowerblue")
ax.set(xlabel='Income level', ylabel='Number of households',
       title='Distribution of household income levels \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{}%".format(int(p.get_height()*100/s)),
            fontsize=8,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(ticks=income_bins, rotation = 25)
fname = statistics_dir + "income.png"
plt.savefig(fname, format='png', dpi=400, bbox_inches='tight')
plt.show()

## Histogram of the number of active workers in households
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="MNPACT", weights="POIDSM", discrete='True', color="cornflowerblue")
ax.set(xlabel='Number of active workers', ylabel='Number of households',
       title='Distribution of the number of active workers within households \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.1f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
fname = statistics_dir + "nworkers.png"
plt.savefig(fname, format='png',dpi=400, bbox_inches='tight')
plt.show()

## Distribution of household type
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="TYPE_MEN", weights="POIDSM",  discrete='True', color="cornflowerblue")
ax.set(xlabel='Household composition', ylabel='Number of households',
       title='Distribution of household types \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.1f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(rotation=18)
fname = statistics_dir + "household_type.png"
plt.savefig(fname, format='png', bbox_inches='tight', dpi=400)
plt.show()


#############################################################################################

## Distribution of public transport modal share
PT_accessibility = Access.loc[:, 'PT_share']
fig, ax = plt.subplots()
ax = sns.kdeplot(x=PT_accessibility, shade=True, color="cornflowerblue")
ax.set(xlabel='Modal share of public transport in communes', ylabel='Share of communes',
       title="Density plot of modal share of public transports in communes")
fname = statistics_dir + "accessibility_communes.png"
plt.savefig(fname, format='png',dpi=400, bbox_inches='tight')
plt.show()

## Distribution of car build date
euro_norms = ['<ECE','Euro 1', 'Euro 2', 'Euro 3', 'Euro 4', 'Euro 5', 'Euro 6']
fig, ax = plt.subplots()
years=[1990, 1993,1997,2001,2006,2011,2016,2020]
ax = sns.histplot(data=Car, x='APMC', weights="POIDSM", bins=years, color="cornflowerblue")
ax.set(xlabel='Year of first entry into service', ylabel='Number of cars',
       title="Distribution of car emission standards \n (EGT 2018-2020)")
fname = statistics_dir + "Euro_norm.png"
s=0
i=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            '{:.2f}%'.format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
    ax.text(p.get_x() + p.get_width() / 2.,
            0,
            euro_norms[i],
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
    i+=1
plt.xticks(ticks=years)
plt.savefig(fname, format='png',dpi=400, bbox_inches='tight')
plt.show()

# Number of cars owned
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="Number of cars owned", weights="POIDSM",  discrete='True', color="cornflowerblue")
ax.set(ylabel='Number of households', title='Distribution of cars owned by households \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(rotation=18)
fname = statistics_dir + "Ncars.png"
plt.savefig(fname, format='png', bbox_inches='tight', dpi=400)
plt.show()

# Number of motorized vehicles owned
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="NB_VEH", weights="POIDSM",  discrete='True', color="cornflowerblue")
ax.set(xlabel='Number of motorized vehicles owned', ylabel='Number of households',
       title='Distribution of vehicles owned by households \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(rotation=18)
fname = statistics_dir + "Nvehicles.png"
plt.savefig(fname, format='png', bbox_inches='tight', dpi=400)
plt.show()

# Number of motorized vehicles owned
fig, ax = plt.subplots()
ax = sns.histplot(data=Household, x="OTHER_VEH", weights="POIDSM",  discrete='True', color="cornflowerblue")
ax.set(xlabel='Number of other motorized vehicles owned', ylabel='Number of households',
       title='Distribution of other vehicles owned by households \n (EGT 2018-2020)')
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.xticks(rotation=18)
fname = statistics_dir + "N_other_vehicles.png"
plt.savefig(fname, format='png', bbox_inches='tight', dpi=400)
plt.show()

########################################################################

# PT share at household level
fig, ax = plt.subplots()
ax = sns.histplot(data=Training_population, x='PT_share_home', weights="POIDSM", bins=10, color="cornflowerblue")
ax.set(xlabel='PT share home', ylabel='Number of households')
fname = statistics_dir + "PT_share_home.png"
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.1f}%".format(float(p.get_height()*100/s)),
            fontsize=8,
            color='black',
            ha='center',
            va='bottom')
plt.title("Distribution of household home accessibility \n (EGT 2018-2020)")
plt.savefig(fname, format='png',dpi=400, bbox_inches='tight')
plt.show()

# Parking
fig, ax = plt.subplots()
ax = sns.histplot(data=Training_population, x='parking', weights="POIDSM", color="cornflowerblue", bins=[0,0.5,1])
ax.set(xlabel='parking', ylabel='Number of households',
       title='Distribution of parking availability for households \n (EGT 2018-2020)')
fname = statistics_dir + "parking.png"
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.savefig(fname, format='png',dpi=400)
plt.show()

# Housing type
fig, ax = plt.subplots()
ax = sns.histplot(data=Training_population, x='housing_type', weights="POIDSM",color="cornflowerblue")
ax.set(xlabel='Housing type', ylabel='Number of households',
       title='Distribution of households housing type \n (EGT 2018-2020)')
fname = statistics_dir + "housing_type.png"
s=0
for p in ax.patches:
    s+= p.get_height()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2.,
            p.get_height(),
            "{:.2f}%".format(float(p.get_height()*100/s)),
            fontsize=10,
            color='black',
            ha='center',
            va='bottom')
plt.savefig(fname, format='png',dpi=400, bbox_inches='tight')
plt.show()
