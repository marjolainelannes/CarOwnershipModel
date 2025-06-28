####################################################################################################
# Purpose: Load public census dataset and calculate the regional accessibility variable
# Note: outputs are car and public transport modal shares per municipality
#       (public transport modal share is taken as proxy for regional accessibility of a municipality)
####################################################################################################

import pandas as pd
import sys
sys.path.append("/include/")
from numeric import get_indexes

##### LOAD DATA AND DEFINE OUTPUT DIRECTORIES ####################################

f_input_data = "../../data/Recensement_MOBPRO_2018.csv"
f_input_var = "../../data/Varmod_MOBPRO_2018.csv"
f_output = "../../data/Accessibility-score.csv" #xls
columns = ["Municipality_code","Municipality_name","Accessibility_score"]

##### READ DATASETS ##############################################################

Var = pd.read_csv(f_input_var, sep= ";")
Data = pd.read_csv(f_input_data, sep= ";")
Data.COMMUNE = Data["COMMUNE"].astype(str)
Data.ARM = Data["ARM"].astype(str)
IDF = ["75","77","78","91","92","93","94","95"]
PT_score_IDF = 0
car_score_IDF = 0
transport_score_IDF = 0

### STEP 1. MERGE "COMMUNE" WITH "ARRONDISSEMENT" #################################

df1 = Var[(Var.COD_VAR == "COMMUNE")]
df2 = Var[(Var.COD_VAR == "ARM")]
Var = pd.concat([df1,df2], ignore_index=True)
print("Number of French people in the database: ", Data.shape[0])
Municipality = list(Var.COD_MOD)
n = len(Municipality)
print("Number of municipalities in France: ", n)

### STEP 2. SCORE THE CAR AND PUBLIC TRANSPORTS MODAL SHARE PER MUNICIPALITY ###########

Columns = ["Municipalities", "M_code",  "PT_share", "Car_share"]
Accessibility = pd.DataFrame(columns=Columns, index=range(0, n))
Data_municipalities = list(Data.COMMUNE)
Data_arr = list(Data.ARM)
for i in range (0,n) :
    m = str(Municipality[i])
    department = m[0:2]
    if (department == "75") & (m != "75056") : # Paris districts
        m_index = get_indexes(Data_arr, m)
    elif (m[0:3] == "132") & (len(m) == 5) : # Marseille districts
        m_index = get_indexes(Data_arr, m)
    elif (m[0:4] == "6938") & (len(m) == 5) : # Lyon districts
        m_index = get_indexes(Data_arr, m)
    else :
        m_index = get_indexes(Data_municipalities, m)
    m_index_Fr = list(Var.COD_MOD).index(m)
    m_name = Var.loc[m_index_Fr, 'LIB_MOD']
    if i % 100 == 0:
        print(m_name)
    PT_score = 0
    car_score = 0
    trans_score = 0
    for j in m_index :
        trans_score = trans_score + float(Data.loc[j,"IPONDI"])
        trans_mode = float(Data.loc[j,"TRANS"])
        if trans_mode == 6 :
            PT_score = PT_score + float(Data.loc[j, "IPONDI"])
        elif trans_mode == 5 :
            car_score = car_score + float(Data.loc[j, "IPONDI"])
    Accessibility.loc[i,"Municipalities"] = m_name #.decode('utf-8')
    Accessibility.loc[i, "M_code"] = m
    if trans_score > 0 :
        PT_share = 100 * PT_score / trans_score
        car_share = 100 * car_score / trans_score
    else :
        PT_share = 0
        car_share = 0
    Accessibility.loc[i, "PT_share"] = PT_share
    Accessibility.loc[i, "Car_share"] = car_share
    # Regional statistics
    if department in IDF :
        PT_score_IDF += PT_score
        car_score_IDF += car_score
        transport_score_IDF += trans_score

PT_share_IDF = 100 * PT_score_IDF / transport_score_IDF
car_share_IDF = 100 * car_score_IDF / transport_score_IDF
Accessibility.loc[n, "Municipalities"] = "Ile-de-France"
Accessibility.loc[n, "M_code"] = "11"
Accessibility.loc[n, "PT_share"] = PT_share_IDF
Accessibility.loc[n, "Car_share"] = car_share_IDF
print("PT_share_IDF: ",PT_share_IDF)
print("car_share_IDF",car_share_IDF)

### SAVE FILE ###########
Accessibility.to_csv(f_output, index = True)