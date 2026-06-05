##################################################################################
# Study: Disaggregated car fleet model
# Purpose: Run training pipeline
# Author: Marjolaine Lannes
# Creation date: December 12, 2021
# Note: Train machine learning models based on EGT survey
##################################################################################
import pandas as pd
from training_functions import *
from MNL_analysis import *
import sys, os
import warnings
warnings.filterwarnings("ignore")

#best_model, training, coefficients, confusion_matrix

# User configuration
input_folder = "C:/Users/marjolaine.lannes/PycharmProjects/Car-choice-prediction/temp_EGT20/training_dataset/"
output_folder = "C:/Users/marjolaine.lannes/PycharmProjects/Car-choice-prediction/temp_EGT20/model_validation/"
predicted_variables = ["N_cars"] # "N_cars","Fuel_type"
plot_odd_ratio = True
assess_best_model = False
logreg_coefficients = False
fix_imbalance = False
logreg_coefficients_smote = False
plots = False
save_model = False
plot_types = ['feature_all','error','confusion_matrix'] # , 'learning', 'parameter'

# Input / Output files
inputs = [input_folder + "Nvehicles.csv", input_folder + "Vehicle_types.csv", input_folder + "Vehicle_types.csv"]
outputs = ["_Ncars.csv", "_Fuel_type.csv", "_Euro_norm.csv"]
legends = ["Number of cars", "Fuel type", "Euro norm"]

# Pipeline
variables = ["N_cars", "Fuel_type", "Euro_norm"]
indexes = [variables.index(item) for item in predicted_variables]
model = ['rf', 'ridge', 'gbc']
for i in indexes :
    predicted_variable = variables[i]
    print(predicted_variable)
    data = pd.read_csv(inputs[i])
    data.drop(columns=["household_id"], inplace=True)
    data = data.replace({'Electric': 'Electric/hybrid', 'Hybrid': 'Electric/hybrid'})
    ### COMPARE MODELS PERFORMANCE ###
    if assess_best_model :
        print("COMPARE MODELS PERFORMANCE")
        foutput_all = output_folder + "compare_all_models" + outputs[i]
        foutput_selected = output_folder + "compare_best_models" + outputs[i]
        best, finalized_models, model_labels = best_model(data, pred=predicted_variable, n_select=14, n_iter=30,
                                                      output_file_all=foutput_all, output_file_tuned=foutput_selected)
        X = get_config('X_train') # data.drop(columns=[predicted_variable])
        y = get_config('y_train') # data[predicted_variable]
        plot_learning_curves(models=finalized_models[0:5], X=X, y=y, labels=model_labels[0:5], cv_folds=10,
                             save_path= output_folder +  "learning_curves" + outputs[i][0:-4])
    ### LOGREG COEFFICIENTS AND LEARNING CURVES ###
    if predicted_variable == "N_cars":
        labels = ['0', '1', '2+']
        models_learning_curves = ['rf','lr', 'lightgbm']
        labels_learning_curves = ['Random Forest', 'Logistic Regression', 'Light Gradient Boosting Machine']
        cmap = cmc.cm.roma
        print(pd.crosstab(data["parking"], data["housing_type"]))
    elif predicted_variable == "Fuel_type":
        labels = ["Diesel", 'Electric/hybrid', "Petrol"] # ["Diesel", "Electric", "Hybrid","Petrol"]
        models_learning_curves = ['ridge', 'lr', 'ada']
        labels_learning_curves = ['Ridge', 'Logistic Regression', 'Ada Boost']
        cmap = cmc.cm.roma
        print(pd.crosstab(data[predicted_variable], data["N_cars"]))
    elif predicted_variable == "Euro_norm":
        labels = ["<ECE", "Euro-1", "Euro-2", "Euro-3", "Euro-4", "Euro-5", "Euro-6"]
        # labels = ["<ECE", "Euro_norm_1", "Euro_norm_2", "Euro_norm_3", "Euro_norm_4", "Euro_norm_5", "Euro_norm_6"]
        # labels = ["<ECE", "Euro 1", "Euro 2", "Euro 3", "Euro 4", "Euro 5", "Euro 6"]
        models_learning_curves = ['ada', 'rf', 'et']
        labels_learning_curves = ['Ada Boost', 'Random Forest', 'Extra Trees']
        cmap = cmc.cm.roma
        print(pd.crosstab(data[predicted_variable], data["N_cars"]))
    if logreg_coefficients:
        print("LOGREG COEFFICIENTS")
        if not assess_best_model:
            best = model_setup(data, predicted_variable, chosen_model=models_learning_curves[0], n_iters=30)
            final_model = finalize_model(best)
        train_idx = get_config("X_train").index
        y_train = get_config("y_train")
        print("y_train")
        print(y_train.value_counts(normalize=True))
        X_train_mnl, y_train_mnl = rebuild_train_without_normalization(data, predicted_variable, train_idx)
        # if predicted_variable == "Fuel_type":
        #     y_train_mnl = y_train_mnl.replace({'Electric':'Electric/hybrid', 'Hybrid':'Electric/hybrid'})
        #     mnl_labels = ["Diesel", 'Electric/hybrid', "Petrol"]
        path_CM = output_folder + "plot/" + predicted_variable + "/MNL_confusion_matrix"
        inference_df, article_table = refit_mnlogit_inference(X_train_mnl, y_train_mnl, labels, path_CM)
        inference_df.to_csv(output_folder + "MNL_statistics_all" + outputs[i])
        article_table.to_latex(output_folder + "MNL_statistics" + outputs[i][0:-4] + ".tex", escape=False)
        article_table.to_csv(output_folder + "MNL_statistics" + outputs[i])
        plot_mnl_OR_multiclass(inference_df, classes= labels[1:], cmap=cmap,
                               save_path= output_folder + "plot/" + predicted_variable + "/MNL_odds_ratios",
                               legend_title=legends[i])
    if plot_odd_ratio:
        inference_df = pd.read_csv(output_folder + "MNL_statistics_all_" + outputs[i])
        plot_mnl_OR_multiclass(inference_df, classes=labels[1:], cmap=cmap,
                               save_path=output_folder + "plot/" + predicted_variable + "/MNL_odds_ratios",
                               legend_title=legends[i], xlim_inf=0.001, xlim_sup=100000)
    ### SAVE MODEL AND PLOT ANALYSES ####
    if plots or save_model:
        if not assess_best_model and not logreg_coefficients:
            tuned_model, tuner, classif = model_setup(data=data, predicted_variable=predicted_variable,
                                                      chosen_model=model[i], n_iters=30, return_tuner=True)
            print("tuner:", tuner)
        else :
            tuned_model = best
        if save_model :
            save(tuned_model, variables[i], output_folder)
        if plots:
            print("PLOTS")
            assessment_plots(tuned_model, labels, plot_types, output_folder + "plot/" + predicted_variable + '/')
    # Test SMOTE method
    if fix_imbalance :
        foutput_selected = output_folder + "compare_best_models_SMOTE_" + outputs[i]
        best_with_SMOTE, finalized_models, model_labels = best_model_imbalanced(data, pred=predicted_variable, n_select=14, n_iter=30,
                                                output_file_tuned=foutput_selected)
        X = get_config('X_train')  # data.drop(columns=[predicted_variable])
        y = get_config('y_train')  # data[predicted_variable]
        plot_learning_curves(models=finalized_models[0:5], X=X, y=y, labels=model_labels[0:5], cv_folds=10,
                             save_path=output_folder + "learning_curves" + outputs[i][0:-4])
        save(best_with_SMOTE, 'SMOTE_' + variables[i], output_folder)
        assessment_plots(best_with_SMOTE, labels, plot_types,
                         output_folder + "plot/SMOTE_" + predicted_variable + '/')
        if logreg_coefficients_smote:
            print("LOGREG COEFFICIENTS")
            if not assess_best_model:
                best = model_setup(data, predicted_variable, chosen_model=models_learning_curves[0], n_iters=30)
                final_model = finalize_model(best)
            train_idx = get_config("X_train").index
            y_train = get_config("y_train")
            print("y_train")
            print(y_train.value_counts(normalize=True))
            X_train_mnl, y_train_mnl = rebuild_train_without_normalization(data, predicted_variable, train_idx)
            path_CM = output_folder + "plot/SMOTE_" + predicted_variable + "/MNL_confusion_matrix"
            inference_df, article_table = refit_mnlogit_inference(X_train_mnl, y_train_mnl, labels, path_CM)
            inference_df.to_csv(output_folder + "SMOTE_MNL_statistics_all" + outputs[i])
            article_table.to_latex(output_folder + "SMOTE_MNL_statistics" + outputs[i][0:-4] + ".tex", escape=False)
            article_table.to_csv(output_folder + "SMOTE_MNL_statistics" + outputs[i])
            plot_mnl_OR_multiclass(inference_df, classes=labels[1:], cmap=cmap,
                                   save_path=output_folder + "plot/SMOTE_" + predicted_variable + "/MNL_odds_ratios",
                                   legend_title=legends[i])
    i += 1
    print("*****************************************************************************")