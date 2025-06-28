##################################################################################
# Purpose: Some utils
##################################################################################

# Function counting the occurences of an element out of a list
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

# Convert in decimals
def decimal_converter(value):
    try:
        return float(str(value).replace(',', '.'))
    except ValueError:
        return value

# Function to get all indexes of an element out of a list
def get_indexes(my_list, e):
    output=[]
    for idx,element in enumerate(my_list) :
        if element == e :
            output.append(idx)
    return(output)
