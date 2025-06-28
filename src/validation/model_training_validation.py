##################################################################################
# Purpose: Run training pipeline
# Note: Train machine learning models based on EGT survey
##################################################################################
import os, sys
sys.path.append("/validation/")
from training_functions import *

# User configuration
input_folder = "../../training_datasets/"
output_folder = "../../output/model_validation/"
predicted_variables = ["N_cars","Fuel_type","Euro_norm"]
assess_best_model = True
plots = True
logreg_coefficients = True
save_model = True
plot_types = ['feature_all','auc','error','confusion_matrix']

# Input / Output files
inputs = [input_folder + "Nvehicles.csv", input_folder + "Vehicle_types.csv", input_folder + "Vehicle_types.csv"]
outputs = ["_Ncars.csv", "_Fuel_type.csv", "_Euro_norm.csv"]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Pipeline
variables = ["N_cars", "Fuel_type", "Euro_norm"]
indexes = [variables.index(item) for item in predicted_variables]
model = ['lr', 'lr', 'gbc']
for i in indexes :
    predicted_variable = variables[i]
    print(predicted_variable)
    data = pd.read_csv(inputs[i])
    data.drop(columns=["household_id"], inplace=True)
    ### COMPARE MODELS PERFORMANCE ###
    if assess_best_model :
        print("COMPARE MODELS PERFORMANCE")
        foutput = output_folder + "compare_models" + outputs[i]
        best, best_df = best_model(data, pred=predicted_variable, output_file=foutput)
        print("best model:", best)
    ### LOGREG COEFFICIENTS ###
    if predicted_variable == "N_cars":
        labels = ['0', '1', '2+']
    elif predicted_variable == "Fuel_type":
        labels = ["Diesel", "Electric", "Hybrid","Petrol"]
    elif predicted_variable == "Euro_norm":
        labels = ["<ECE", "Euro 1", "Euro 2", "Euro 3", "Euro 4", "Euro 5", "Euro 6"]
    if logreg_coefficients:
        lr_coef = True
        print("LOGREG COEFFICIENTS")
        foutput = output_folder + "coefficients" + outputs[i][0:-4]
        lr_model = model_setup(data=data, predicted_variable=predicted_variable, chosen_model='lr', n_iters=50)
        coefficients(tuned_lr_model=lr_model, predicted_variable = predicted_variable, labels = labels, output_file=foutput)
    ### SAVE MODEL AND PLOT ANALYSES ####
    if plots or save_model:
        if assess_best_model:
            tuned_model, tuner, classif = model_setup(data=data, predicted_variable=predicted_variable, chosen_model=best,
                                             n_iters=50, return_tuner=True)
        else:
            tuned_model, tuner, classif = model_setup(data=data, predicted_variable=predicted_variable,
                                             chosen_model=model[i], n_iters=50, return_tuner=True)
        name = 'pipeline_' + variables[i]
        if save_model :
            print("tuner:", tuner)
            save(tuned_model, name)
        if plots:
            print("PLOTS")
            assessment_plots(tuned_model, labels, plot_types)
    i += 1