##################################################################################
# Study: Disaggregated car fleet model
# Purpose: Define training functions
# Author: Marjolaine Lannes
# Creation date: December 12, 2021
# Note: Train machine learning models based on EGT survey
##################################################################################
URI = "your/path/"
import pandas as pd
import mlflow
mlflow.set_tracking_uri(URI)
import mlflow.sklearn
from pycaret.classification import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

def best_model(data:pd.DataFrame, pred:str, output_file_all:str, output_file_tuned:str, n_select:int = 5, n_iter: int = 30) : #, weights
    categorical_features_list = ["age", "N_workers", "household_type", "housing_type", "parking", "parking_at_workplace"]
    ordinal_features_list = {'N_cars':['0','1','2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    ## setup data and tune model ##
    if pred == 'N_cars':
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features={}, memory= False, system_log= False,
                    categorical_features = categorical_features_list, numeric_features=numeric_features_list)
    elif pred == 'Fuel_type':
        data.drop(columns=['Euro_norm'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features=ordinal_features_list, memory= False, system_log= False,
                    categorical_features = categorical_features_list, numeric_features=numeric_features_list)
    elif pred == 'Euro_norm':
        print(data.columns)
        data.drop(columns=['Fuel_type'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax',
                    max_encoding_ohe=-1, ordinal_features=ordinal_features_list, numeric_features=numeric_features_list)
    ## evaluate models and compare models ##
    print("Comparing baseline models...")
    top_models = compare_models(sort='MCC', n_select=n_select)
    all_results = pull()
    all_results.to_csv(output_file_all)
    print("Tuning top models...")
    tuned_models = []
    for model in top_models:
        print(model)
        tuned = tune_model(model, optimize="MCC", n_iter=n_iter, choose_better=True)
        tuned_models.append(tuned)
    print("Comparing tuned models...")
    best_tuned_model = compare_models(include=tuned_models, sort="MCC")
    ordered_models = compare_models(include=tuned_models, sort="MCC", n_select=n_select)
    model_labels = []
    for model in ordered_models:
        model_labels.append(type(model).__name__)
    ## report the best model ##
    ordered_tuned_models = pull()
    ordered_tuned_models.to_csv(output_file_tuned)
    final_models = [finalize_model(m) for m in tuned_models] # ordered_tuned_models???
    print(model_labels)
    return(best_tuned_model, final_models, model_labels)

def best_model_imbalanced(data:pd.DataFrame, pred:str, output_file_tuned:str, n_select:int = 5,
                          n_iter: int = 30, fix_method = 'SMOTE') : #, weights
    categorical_features_list = ["age", "N_workers", "household_type", "housing_type", "parking", "parking_at_workplace"]
    categorical_features_list_cars = categorical_features_list + ['N_cars']
    ordinal_features_list = {'N_cars':['0','1','2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    ## setup data and tune model ##
    if pred == 'N_cars':
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1, ordinal_features={},
                    memory= False, system_log= False,  categorical_features = categorical_features_list,
                    numeric_features=numeric_features_list, fix_imbalance=True, fix_imbalance_method=fix_method)
    elif pred == 'Fuel_type':
        if 'Euro_norm' in data.columns:
            data.drop(columns=['Euro_norm'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features=ordinal_features_list, memory= False, system_log= False,
                    categorical_features = categorical_features_list_cars, numeric_features=numeric_features_list,
                    fix_imbalance=True, fix_imbalance_method=fix_method)
    elif pred == 'Euro_norm':
        print(data.columns)
        if 'Fuel_type' in data.columns:
            data.drop(columns=['Fuel_type'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features=ordinal_features_list, memory= False, system_log= False,
                    categorical_features = categorical_features_list_cars, numeric_features=numeric_features_list,
                    fix_imbalance=True, fix_imbalance_method=fix_method)
    ## evaluate models and compare models ##
    print("Comparing baseline models...")
    top_models = compare_models(sort='MCC', n_select=n_select)
    print("Tuning top models...")
    tuned_models = []
    for model in top_models:
        print(model)
        tuned = tune_model(model, optimize="MCC", n_iter=n_iter, choose_better=True)
        tuned_models.append(tuned)
    print("Comparing tuned models...")
    best_tuned_model = compare_models(include=tuned_models, sort="MCC")
    model_labels = []
    ordered_models = compare_models(include=tuned_models, sort="MCC", n_select=n_select)
    for model in ordered_models:
        model_labels.append(type(model).__name__)
    ## report the best model ##
    ordered_tuned_models = pull()
    ordered_tuned_models.to_csv(output_file_tuned)
    final_models = [finalize_model(m) for m in ordered_models]
    print(model_labels)
    return (best_tuned_model, final_models, model_labels)

def plot_learning_curves(models, X, y, labels:list, cv_folds=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                         save_path="learning_curve.png"):
    """
    Draw the learning curves for several models with MCC and ± std.
    Args:
        models : list of estimators
        X : features (DataFrame ou ndarray)
        y : target (Series ou ndarray)
        labels : list de noms des modèles
        cv_folds : nombre de folds CV
        n_jobs : parallel jobs
        train_sizes : share of dataset for the learning curve
        save_path : path
    """
    colors = plt.cm.tab10.colors
    for score in ["Accuracy", "MCC"]:
        fig, ax= plt.subplots()
        if score == "MCC":
            scorer = make_scorer(matthews_corrcoef)
        else :
            scorer = "accuracy"
        # n_samples = X.shape[0]
        for idx, (model, label) in enumerate(zip(models, labels)):
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=clone(model), X=X, y=y, train_sizes=train_sizes, scoring=scorer, n_jobs=n_jobs,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            # Plot
            color = colors[idx % len(colors)]
            ax.plot(train_sizes_abs, train_mean, 'o-', color = color, label=f"{label} - train")
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
            ax.plot(train_sizes_abs, test_mean, 's--', color = color, label=f"{label} - validation")
            ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
        ax.set_xlabel("Training set size")
        ax.set_ylabel(score)
        ax.set_title("Learning Curves")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}_{score}.png", dpi=300)

def tune_and_plot_learning_curves(data, pred, models_names:list,labels:list, cv_folds=5, n_jobs=-1, n_iter=30,
                                  train_sizes=np.linspace(0.1, 1.0, 10), save_path="learning_curves"):
    ## setup data and tune models ##
    categorical_features_list = ["age", "N_workers", "household_type", "housing_type", "parking",
                                 "parking_at_workplace"]
    categorical_features_list_cars = categorical_features_list + ['N_cars']
    ordinal_features_list = {'N_cars': ['0', '1', '2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    if pred == 'N_cars':
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features={}, memory=False, system_log=False,
                    categorical_features=categorical_features_list, numeric_features=numeric_features_list)
    elif pred == 'Fuel_type':
        data.drop(columns=['Euro_norm'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features=ordinal_features_list, memory=False, system_log=False,
                    categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list)
    elif pred == 'Euro_norm':
        data.drop(columns=['Fuel_type'], inplace=True)
        exp = setup(data, target=pred, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                    ordinal_features=ordinal_features_list, memory=False, system_log=False,
                    categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list)
    models = []
    for model_name in models_names:
        trained_model = create_model(model_name, return_train_score=True)
        tuned_model = tune_model(trained_model, optimize="MCC", n_iter=n_iter, choose_better=True)
        models.append(tuned_model)
    # plot learning curves
    X = data.drop(columns=[pred])
    y = data[pred]
    colors = cm.get_cmap('tab10', len(models))
    for score in ["Accuracy", "MCC"]:
        plt.figure(figsize=(10, 7))
        if score == "MCC":
            scorer = make_scorer(matthews_corrcoef)
        else :
            scorer = "accuracy"
        # n_samples = X.shape[0]
        for idx, (model, label) in enumerate(zip(models, labels)):
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=clone(model), X=X, y=y, train_sizes=train_sizes, scoring=scorer, n_jobs=n_jobs,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            # Plot
            plt.plot(train_sizes_abs, train_mean, 'o-', color = colors(idx), label=f"{label} - train")
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.plot(train_sizes_abs, test_mean, 's--', color = colors(idx), label=f"{label} - validation")
            plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel("Training set size")
        plt.ylabel(score)
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path + "_" + score + '.png', dpi=300)

def model_setup(data:pd.DataFrame, predicted_variable:str, chosen_model:str, n_iters:int, return_tuner = False) :
    categorical_features_list = ["age", "N_workers", "household_type", "parking", "parking_at_workplace",
                                 "housing_type"]
    categorical_features_list_cars = categorical_features_list + ['N_cars']
    ordinal_features_list = {'N_cars': ['0', '1', '2+']}
    numeric_features_list = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    ## setup data and tune model ##
    if predicted_variable == 'N_cars':
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                         ordinal_features={}, memory= False, system_log= False,
                         categorical_features=categorical_features_list, numeric_features=numeric_features_list,
                         log_experiment=True, experiment_name=f"{predicted_variable}_{chosen_model}", log_plots=True)
    elif predicted_variable == 'Fuel_type':
        if 'Euro_norm' in data.columns:
            data.drop(columns=['Euro_norm'], inplace=True)
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                         categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list,
                         ordinal_features=ordinal_features_list, memory= False, system_log= False,
                         log_experiment=True, experiment_name=f"{predicted_variable}_{chosen_model}", log_plots=True)
    elif predicted_variable == 'Euro_norm':
        if 'Fuel_type' in data.columns:
            data.drop(columns=['Fuel_type'], inplace=True)
        classif1 = setup(data, target=predicted_variable, normalize=True, normalize_method='minmax', max_encoding_ohe=-1,
                         categorical_features=categorical_features_list_cars, numeric_features=numeric_features_list,
                         ordinal_features=ordinal_features_list, memory= False, system_log= False,
                         log_experiment=True, experiment_name=f"{predicted_variable}_{chosen_model}", log_plots=True)
    print("setup done")
    model = create_model(chosen_model, return_train_score = True)
    print("model created")
    ## tune model hyperparameters ##
    tuned_model, tuner = tune_model(model, choose_better=True,n_iter=n_iters, optimize = "MCC", return_tuner=True) #n_iter
    print("model tuned")
    if return_tuner :
        return(tuned_model, tuner, classif1)
    else :
        return(tuned_model)

def assessment_plots(tuned_model,labels, plots:list, output_path:str) :
    for figure in plots :
        print(figure)
        if figure == 'confusion_matrix' :
            plot_model(tuned_model, plot=figure, save = output_path,
                       plot_kwargs = {'percent': True, 'labels':labels},scale=20)
        elif figure == 'SHAP':
            interpret_model(tuned_model, save=output_path + 'SHAP.png')
        else :
            plot_model(tuned_model, plot=figure,scale=20, save = output_path)

def save(tuned_model, model_name, output_path):
    print('Hold-out test ')
    predict_model(tuned_model)
    print('Finalize model')
    final_model = finalize_model(tuned_model)
    print('SAVE :', final_model)
    save_model(final_model, output_path + model_name, model_only = True)
    # save with MLflow
    mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model")