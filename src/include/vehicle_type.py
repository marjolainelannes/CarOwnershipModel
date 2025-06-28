##################################################################################
# Purpose: Functions converting year of first entry into service in Euro Norm
# Note: the output is the European emission standard of the car
##################################################################################

def euro_norm_construction(month,year):
    if (year<1996) or (year == 1996 and month < 7) :  # entre le 1er janvier 1993 et le 1er juillet 1996
        norm = 1 #'Euro 1'
    elif (year == 1996 and month >= 7) or 1996 < year < 2001 : # entre le 1er juillet 1996 et le 1er janvier 2001
        norm = 2 # 'Euro 2'
    elif 2001 <= year < 2006 : # entre le 1er janvier 2001 et le 1er janvier 2006
        norm = 3 # 'Euro 3'
    elif 2006 <= year < 2011 :  # du 1er janvier 2006 au 1er janvier 2011
        norm = 4 # 'Euro 4'
    elif 2011 <= year < 2015 or (year == 2015 and month < 9) :  # du 1er janvier 2011 au 1er septembre 2015
        norm = 5 # 'Euro 5'
    elif (year == 2015 and month >= 9) or 2015 < year < 2018 or (year == 2018 and month < 9) :  # du 1er septembre 2015 au 1er septembre 2018
        norm = 6 # 'Euro 6b'
    elif (year == 2018 and month >= 9) or (year == 2019 and month < 9) :  # du 1er septembre 2018 au 1er septembre 2019
        norm = 7 # 'Euro 6c'
    elif (year == 2019 and month >= 9) or 2019 < year < 2021 :  # du 1er septembre 2019 au 1er janvier 2021
        norm = 8 # 'Euro 6d-TEMP'
    elif 1 < year < 1993 :
        norm = 0 # "Antérieur à Euro 1"
    return(norm)

def emission_standard(year) :
    if 1 < year < 1993: # avant le 1er janvier 1993
        norm = "<ECE"
    elif 1993 <= year < 1997 : # entre le 1er janvier 1993 et  le 1er juillet 1996
        norm = 'Euro-1'
    elif 1997 <= year < 2001 : # entre le 1er juillet 1996 et le 1er janvier 2001
        norm = 'Euro-2'
    elif 2001 <= year < 2006 : # entre le 1er janvier 2001 et le 1er janvier 2006
        norm = 'Euro-3'
    elif 2006 <= year < 2011 :  # du 1er janvier 2006 au 1er janvier 2011
        norm = 'Euro-4'
    elif 2011 <= year < 2016 :  # du 1er janvier 2011 au 1er septembre 2015
        norm = 'Euro-5'
    elif 2016 <= year :  # à partir du 1er septembre 2015
        norm = 'Euro-6'  # = 'Euro 6b', 'Euro 6c' et 'Euro 6d-TEMP'
    return(norm)
