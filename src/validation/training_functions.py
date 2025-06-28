##################################################################################
# Purpose: Define training functions
# Note: Train machine learning models based on EGT survey
##################################################################################
import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns

def best_model(data:pd.DataFrame, pred:str, output_file:str) :
    categorical_features_list = ["age", "N_workers", "household_type", "housing_type", "parking", "parking_at_workplace"]
    categorical_features_list_cars = categorical_features_list + ['N_cars']
    ordinal_features_list = {'N_cars':['0','1','2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    ## setup data and tune model ##
    if pred == 'N_cars':
        classif1 = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1, ordinal_features={}, memory= False, system_log= False,
                         categorical_features = categorical_features_list, numeric_features=numeric_features_list)
    elif pred == 'Fuel_type':
        data.drop(columns=['Euro_norm'], inplace=True)
        classif1 = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1, ordinal_features=ordinal_features_list, memory= False, system_log= False,
                         categorical_features = categorical_features_list_cars, numeric_features=numeric_features_list)
    elif pred == 'Euro_norm':
        data.drop(columns=['Fuel_type'], inplace=True)
        classif1 = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,ordinal_features=ordinal_features_list, memory= False, system_log= False,
                         categorical_features = categorical_features_list_cars, numeric_features=numeric_features_list)
    ## evaluate models and compare models ##
    best = compare_models(sort='MCC')
    ## report the best model ##
    best_results = pull()
    best_results.to_csv(output_file)
    return(best, best_results)

def model_setup(data:pd.DataFrame, predicted_variable:str, chosen_model:str, n_iters:int, return_tuner = False) :
    categorical_features_list = ["age", "N_workers", "household_type", "parking", "parking_at_workplace",
                                 "housing_type"]
    categorical_features_list_cars = categorical_features_list + ['N_cars']
    ordinal_features_list = {'N_cars': ['0', '1', '2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    ## setup data and tune model ##
    if predicted_variable == 'N_cars':
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,ordinal_features={}, memory= False, system_log= False,
                         categorical_features=categorical_features_list, numeric_features=numeric_features_list)
    elif predicted_variable == 'Fuel_type':
        if 'Euro_norm' in data.columns:
            data.drop(columns=['Euro_norm'], inplace=True)
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                         categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list,
                         ordinal_features=ordinal_features_list, memory= False, system_log= False,)
    elif predicted_variable == 'Euro_norm':
        if 'Fuel_type' in data.columns:
            data.drop(columns=['Fuel_type'], inplace=True)
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                         categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list,
                         ordinal_features=ordinal_features_list, memory= False, system_log= False)
    model = create_model(chosen_model, return_train_score = True)
    ## tune model hyperparameters ##
    tuned_model, tuner = tune_model(model, choose_better=True,n_iter=n_iters, optimize = "MCC", return_tuner=True)
    if return_tuner :
        return(tuned_model, tuner, classif1)
    else :
        return(tuned_model)

def assessment_plots(tuned_model,labels, plots:list) : #, classif
    for figure in plots :
        if figure == 'confusion_matrix' :
            plot_model(tuned_model, plot=figure, plot_kwargs = {'percent': True, 'labels':labels}, save=True,scale=5)
        elif figure == 'SHAP':
            interpret_model(tuned_model, save=True)
        else :
            plot_model(tuned_model, plot=figure, save = True,scale=5)

def coefficients(tuned_lr_model, predicted_variable, labels, output_file):
    ## get coefficients ##
    X_train = get_config('X_train')
    features = tuned_lr_model.feature_names_in_
    coef_df = pd.DataFrame(data = tuned_lr_model.coef_, columns = features, index=tuned_lr_model.classes_).transpose()
    coef_df.index = coef_df.index.set_names(['features'])
    coef_df.reset_index(inplace=True)
    ## plot graph of coefficients ##
    Coef_graph = pd.DataFrame(columns=['features', predicted_variable, 'coefficients'])
    n_features = len(features)
    n_labels = len(labels)
    for i in range (0, n_features) :
        for j in range (0, n_labels) :
            new = [[coef_df.iloc[i,0], labels[j], coef_df.iloc[i,j+1]]]
            new_row = pd.DataFrame(new, columns = ['features', predicted_variable, 'coefficients'])
            Coef_graph = pd.concat([Coef_graph, new_row], ignore_index=True)
    sns.catplot(data=Coef_graph, x='coefficients', y='features', hue=predicted_variable, kind='bar',orient ="h")
    plt.savefig(output_file+'.png', dpi=400)
    plt.show()
    coef_df.to_csv(output_file+'.csv')

def save(tuned_model, model_name):
    final_model = finalize_model(tuned_model)
    print('SAVE :', final_model)
    save_model(final_model, model_name=model_name, model_only = True)
