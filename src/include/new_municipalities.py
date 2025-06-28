##################################################################################
# Purpose: Function converting municipalities id from 2018 to 2020
# Note: data collected from Insee (French national statistics institute)
##################################################################################

def new_municipalities(municipality):
    if municipality == 77491 :
        municipality = 77316 ## Veneux-les-Sablons (77491) a intégré la commune de Moret-et-Orvanne (77316) entre 2018 et 2020
    if municipality == 77028 :
        municipality = 77433 ## Beautheil-Saints (77433) remplace Beautheil (77028) et Saints (77433) en 2019
    if municipality == 91182 :
        municipality = 91228 ## Évry-Courcouronnes (91228) remplace Courcouronnes (91182) et Évry (91228) en 2019
    if municipality == 78251 :
        municipality = 78551 ## Fourqueux devient commune déléguée au sein de Saint-Germain-en-Laye (78551) en 2019
    if municipality == 78503 :
        municipality = 78320 ## Notre-Dame-de-la-Mer (78320) remplace Jeufosse (78320) et Port-Villez (78503) en 2019
    if municipality == 78524 :
        municipality = 78158 ## Chesnay-Rocquencourt (78158) remplace Le Chesnay (78158) et Rocquencourt (78524) en 2019
    return(municipality)

def new_municipality_name(municipality):
    if municipality == "Beautheil" or municipality == "Saints":
        municipality = "Beautheil-Saints" ## Beautheil-Saints (77433) remplace Beautheil (77028) et Saints (77433) en 2019
    if municipality == "Courcouronnes" or municipality == "Évry":
        municipality = "Evry-Courcouronnes" ## Évry-Courcouronnes (91228) remplace Courcouronnes (91182) et Évry (91228) en 2019
    if municipality == "Fourqueux" :
        municipality = "Saint-Germain-en-Laye" ## Fourqueux devient commune déléguée au sein de Saint-Germain-en-Laye (78551) en 2019
    if municipality == "Jeufosse" or municipality == "Port-Villez" :
        municipality = "Notre-Dame-de-la-Mer" ## Notre-Dame-de-la-Mer (78320) remplace Jeufosse (78320) et Port-Villez (78503) en 2019
    if municipality == "Le Chesnay" or municipality == "Rocquencourt":
        municipality = "Le Chesnay-Rocquencourt"   ## Chesnay-Rocquencourt (78158) remplace Le Chesnay (78158) et Rocquencourt (78524) en 2019
    return(municipality)