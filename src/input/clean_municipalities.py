##################################################################################
# Purpose: Clean input data (EGT 2018-2020) before preprocessing
# Note: Municipalities are not formatted for work / study municipalities in EGT
##################################################################################
import pandas as pd
import numpy as np
import sys
sys.path.append("/include/")
from new_municipalities import new_municipality_name

# Input / output data
data_dir = "../../data/"
individuals_f = data_dir +  "EGT20/02_Individu_egt1820.xlsx"
travels_f = data_dir +  "EGT20/03_Deplacement_egt1820.xlsx"
accessibility_f = data_dir + "Accessibility-score-Fr.xlsx"
output_f = data_dir + "cleaned_individuals.csv"

# Load data
Individuals = pd.read_excel(individuals_f)
Travels = pd.read_excel(travels_f)
n_ind = Individuals.shape[0]
Accessibility = pd.read_excel(accessibility_f)

# Filter and clean cities
names_dict = {'VERNEUIL EN HALATTE (60)':'Verneuil-en-Halatte', 'INCARVILLE (27)':'Incarville',"Herblay":"Herblay-sur-Seine",
             "AUBEVOYE (27)":"Le Val d'Hazey", "GELLAINVILLE (28)":"Gellainville", "CROUY EN THELLE (60)":"Crouy-en-Thelle",
             "TOULON (83)":"Toulon","PACY SUR EURE (27)":"Pacy-sur-Eure", "ORLEANS (45)":"Orléans","République":"Paris 11e",
             "Rigollots, Roublot, Carrières":"Fontenay-sous-Bois", "Verneuil-l'Étang":"Verneuil-l'Etang",
              "Herblay (95220)":"Herblay-sur-Seine", "Saint-Cyr-l'École":"Saint-Cyr-l'Ecole", "Paris ":"Paris",
              "MERU (60)":"Méru", "CREIL (60)":"Creil", "Chantiers":"Versailles", "Bois Cadet, Montesquieu":"Fontenay-sous-Bois",
              "Monmousseau - Vérollot":"Ivry-sur-Seine", "Echat":"Créteil","Louis Bertrand - Mirabeau - Sémard":"Ivry-sur-Seine",
              "Village Rueil sur Seine":"Rueil-Malmaison"}
quartiers_dict = {"Quartier de la Porte-Saint-Denis":"Paris 10e", "Quartier Saint-Georges":"Paris 9e", "Quartier des Épinettes":"Paris 17e",
                  "Quartier de Clignancourt":"Paris 18e","Quartier de la Sorbonne":"Paris 5e","Quartier de la Folie-Méricourt":"Paris 11e",
                  "Quartier du Père-Lachaise":"Paris 20e","Quartier des Grandes-Carrières":"Paris 18e","Quartier Notre-Dame-des-Champs":"Paris 6e",
                  "Quartier de la Chaussée-d'Antin":"Paris 9e", "Quartier du Mail":"Paris 2e","Quartier du Parc Nord":"Nanterre"}
names_dict.update(quartiers_dict)
Individuals = Individuals.replace(names_dict)

# remove INSEE code from municipality name in Accessibility table
for m, municipality in enumerate(Accessibility['Municipalities']):
    municipality_name = municipality[:-8]
    Accessibility.loc[m,'Municipalities'] = municipality_name

# communes
departements = {'Val-de-Marne':94028, 'Yvelines':78646, 'Hauts-de-Seine':92050,'Seine-Saint-Denis':93008,"Val-d'Oise":95500,'Seine-et-Marne':77288,'Essonne':91228}
communes = list(Accessibility.Municipalities)
places_not_cities = []
for person in range(0,n_ind):
    person_id = Individuals.loc[person, "IDCEREMA"]
    working_city = Individuals.loc[person, "TRAVCOMM"]
    study_city = Individuals.loc[person, "ETUDCOMM"]
    # Work
    if not pd.isnull(working_city):
        # erase INSEE municipality code
        if working_city[-1] == ")":
            n = working_city.index("(") - 1
            working_city = working_city[0:n]
        # in Paris, if the arrondissement is given, check format
        if working_city[0:5] == "Paris" :
            if len(working_city) > 5 and len(working_city) < 15 :
                working_city = working_city + " Arrondissement"
        # name format
        if working_city[0] == "É" :
            working_city = "E" + working_city[1:]
        working_city = new_municipality_name(working_city)
        # if working city is not a municipality, search for additional information
        if working_city not in communes:
            places_not_cities.append(working_city)
            indiv_travels = Travels[Travels.IDCEREMA == person_id].reset_index()
            travel_motives_departure = list(indiv_travels.loc[:,"ORMOT_H9"])
            travel_motives_arrival = list(indiv_travels.loc[:, "DESTMOT_H9"])
            if 2 in travel_motives_departure: # i.e. if one of the registered travels is commuting
                i = travel_motives_departure.index(2)
                if len(str(indiv_travels.loc[i, "ORINSEE"])) > 2:
                    working_city_code = int(indiv_travels.loc[i,"ORINSEE"])
            elif 2 in travel_motives_arrival: # i.e. if one of the registered travels is commuting
                i = travel_motives_arrival.index(2)
                if len(str(indiv_travels.loc[i,"DESTINSEE"])) > 2 :
                    working_city_code = int(indiv_travels.loc[i,"DESTINSEE"])
            else :
                working_city_code = Individuals.loc[person, "RESCOMM"]
                if working_city in departements.keys() :
                    dept_code = departements[working_city]
                    if str(working_city)[0:2] != str(dept_code)[0:2]  :
                        working_city_code = dept_code
                print("ERROR ", person_id, working_city, working_city_code)
                working_city_index = list(Accessibility.M_code).index(int(working_city_code))
        else:
            working_city_index = communes.index(str(working_city))
            working_city_code = Accessibility.loc[working_city_index, "M_code"]
        working_city_accessibility = Accessibility.loc[working_city_index, "PT_share"]
    else :
        working_city_code = np.nan
        working_city_accessibility = np.nan
    # Study
    if not pd.isnull(study_city):
        # erase INSEE municipality code
        if study_city[-1] == ")":
            n = study_city.index("(") - 1
            study_city = study_city[0:n]
        # in Paris, if the arrondissement is given, check format
        if study_city[0:5] == "Paris":
            if len(study_city) > 5 and len(study_city) < 15 :
                study_city = study_city + " Arrondissement"
        # name format
        if study_city[0] == "É":
            study_city = "E" + study_city[1:]
        study_city = new_municipality_name(study_city)
        # if study city is not a municipality, search for additional information
        if study_city not in communes :
            places_not_cities.append(study_city)
            indiv_travels = Travels[Travels.IDCEREMA == person_id].reset_index()
            travel_motives_departure = list(indiv_travels.loc[:, "ORMOT_H9"])
            travel_motives_arrival = list(indiv_travels.loc[:, "DESTMOT_H9"])
            if 4 in travel_motives_departure:  # i.e. if one of the registered travels is studying
                i = travel_motives_departure.index(4)
                if len(str(indiv_travels.loc[i, "ORINSEE"])) > 2:
                    study_city_code = int(indiv_travels.loc[i, "ORINSEE"])
            elif 4 in travel_motives_arrival:  # i.e. if one of the registered travels is studying
                i = travel_motives_arrival.index(4)
                if len(str(indiv_travels.loc[i, "DESTINSEE"])) > 2:
                    study_city_code = int(indiv_travels.loc[i, "DESTINSEE"])
            else:
                study_city_code = Individuals.loc[person, "RESCOMM"]
                if study_city in departements.keys() :
                    dept_code = departements[study_city]
                    if str(study_city)[0:2] != str(dept_code)[0:2]  :
                        study_city_code = dept_code
                print("ERROR ", person_id, study_city, study_city_code)
                study_city_index = list(Accessibility.M_code).index(study_city_code)
        else:
            study_city_index = communes.index(study_city)
            study_city_code = Accessibility.loc[study_city_index, "M_code"]
        study_city_accessibility = Accessibility.loc[study_city_index, "PT_share"]
    else :
        study_city_code = np.nan
        study_city_accessibility = np.nan
    # Update
    updated_working_city = np.nanmax([working_city_code, study_city_code])
    accessibility_score = np.nanmax([working_city_accessibility, study_city_accessibility])
    Individuals.loc[person, "PT_SHARE_WORK"] = accessibility_score

# Save data
Individuals = Individuals[['IDCEREMA','PKVPTRAV','TRAGE','DDOMTRAV','PT_SHARE_WORK']]
Individuals.to_csv(output_f, index=False)
places_not_cities = list(set(places_not_cities))
print(places_not_cities)