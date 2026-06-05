from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def get_confusion_matrix(result, X_train, y_train, save_path):
    X_const = sm.add_constant(X_train)
    pred_probs = result.predict(X_const)
    y_pred = pred_probs.idxmax(axis=1)
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize="true", cmap="Greens", values_format=".1f")
    plt.title("Multinomial Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.show()

def MNL_results(result, other_classes, decimals = 3, line_break=False, include_fit_stats=True):
    print(result.summary())
    params = result.params
    pvals = result.pvalues
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    stars = pvals.applymap(significance_stars)
    bse = result.bse
    summary_table = pd.DataFrame({"variable": params.index}, columns=["variable"])
    article_table = pd.DataFrame(index=params.index)
    if type(params.columns[-1]) == int :
        class_labels = [other_classes[class_number] for class_number in params.columns]
    else :
        class_labels = params.columns
    for c, class_label in enumerate(class_labels):
        ci_lower = []
        ci_upper = []
        for var in params.index:
            lower = conf.loc[(class_label, var), "lower"]
            upper = conf.loc[(class_label, var), "upper"]
            ci_lower.append(np.exp(lower))
            ci_upper.append(np.exp(upper))
        summary_class = pd.DataFrame({
            "variable": params.index,
            "coef": params[c],
            "odds_ratio": odds_ratios[c],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": pvals[c],
            "stars": stars[c]
        })
        new_cols = []
        for col in summary_class.columns:
            if (col != "variable"):
                new_cols.append(f"{col}_{class_label}")
            else:
                new_cols.append(col)
        summary_class.columns = new_cols
        summary_table = pd.merge(summary_table, summary_class, how='left', on="variable")

        # Formatting for article
        coef = params[c]
        se = bse[c]
        t_stat = coef / se
        star = stars[c]
        if not line_break: # Format: coef*** (t)
            formatted = [f"{coef.iloc[i]:.{decimals}f}{star.iloc[i]} " 
                         f"({t_stat.iloc[i]:.{decimals}f})" for i in range(len(coef))]
            article_table[class_label] = formatted
        else: # Format: coef*** \n (t)
            col_values = []
            for i in range(len(coef)):
                col_values.append(f"{coef.iloc[i]:.{decimals}f}{star.iloc[i]}")
                col_values.append(f"({t_stat.iloc[i]:.{decimals}f})")
            article_table[class_label] = col_values
    if line_break: # Duplicate index for double-line layout
        new_index = []
        for var in params.index:
            new_index.append(var)
            new_index.append("")
        article_table.index = new_index
    article_table = article_table.rename(index={"const": "Intercept"})
    if include_fit_stats:
        # McFadden R²
        r2_mcfadden = result.prsquared
        n_obs = int(result.nobs)
        article_table.loc[""] = [""] * article_table.shape[1]
        article_table.loc["Observations"] = [f"{n_obs}"] + [""] * (article_table.shape[1] - 1)
        article_table.loc["McFadden R²"] = [f"{r2_mcfadden:.3f}"] + [""] * (article_table.shape[1] - 1)
    return summary_table, article_table

def plot_mnl_OR_multiclass(inference_df, classes, cmap, save_path, legend_title="Class vs reference", figsize=(8,10),
                           set_xlim=True, xlim_inf = 0.01, xlim_sup=100):
    # Prepare data
    df = inference_df.copy()
    df = df[df["variable"] != "const"]
    variables = df["variable"].unique()
    n_classes = len(classes)
    colors = [cmap(i/(n_classes-1)) for i in range(n_classes)]
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(variables))
    for i, cls in enumerate(classes):
        ax.errorbar(df[f"odds_ratio_{cls}"], y_positions + (i - n_classes/2)*0.1,
            xerr=[df[f"odds_ratio_{cls}"] - df[f"ci_lower_{cls}"],
                  df[f"ci_upper_{cls}"] - df[f"odds_ratio_{cls}"]],
            fmt='o', label=cls, color=colors[i], alpha=0.9)
    ax.axvline(x=1, linestyle='--', color='grey')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xscale("log")
    if set_xlim:
        ax.set_xlim(xlim_inf, xlim_sup)
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=320)
    plt.show()

def compute_accuracy_metrics(result, X, y):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
    X_const = sm.add_constant(X)
    pred_probs = result.predict(X_const)
    y_pred = pred_probs.idxmax(axis=1)
    # Mcfadden R2
    ll_model = result.llf
    X_null = sm.add_constant(pd.DataFrame(np.ones(len(y)), columns=["intercept"]))
    null_model = sm.MNLogit(y, X_null).fit(disp=False)
    ll_null = null_model.llf
    r2_mcfadden = 1 - (ll_model / ll_null)
    # Overall accuracy
    acc = accuracy_score(y, y_pred)
    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y, y_pred)
    # Per-class accuracy
    report = classification_report(y, y_pred, output_dict=True)
    per_class_accuracy = {cls: report[cls]["recall"] for cls in y.cat.categories}
    return r2_mcfadden, acc, bal_acc, per_class_accuracy

def rebuild_train_without_normalization(data, target, train_idx, drop_first=True):
    categorical_features = ["age", "N_workers", "household_type", "housing_type", "parking", "parking_at_workplace"]
    ordinal_features = {'N_cars': ['0', '1', '2+']} if target != "N_cars" else {}
    numeric_features = ["income", 'PT_share_home', 'PT_share_work', 'commuting_distance']
    df_train = data.loc[train_idx].copy()
    y_train = df_train[target].astype("category")
    # Ordinal encoding
    for var, order in ordinal_features.items():
        df_train[var] = pd.Categorical(df_train[var], categories=order, ordered=True).codes
    # One-hot encoding
    X_train = df_train[numeric_features + categorical_features + list(ordinal_features.keys())].copy()
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=drop_first)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in numeric_features:
        if col in ["income", 'commuting_distance']:
            X_train[col] = X_train[col].fillna(0)
        else:
            X_train[col] = X_train[col].fillna(X_train[col].median()) # X_train[col].median()
    X_train = X_train.fillna(0)
    X_train = X_train.astype(float)
    return X_train, y_train

def refit_mnlogit_inference(X_train, y_train, labels:list, path_confusion_matrix:str):
    reference_class = labels[0]
    other_classes = labels.copy()
    other_classes.remove(reference_class)
    categories = list(y_train.cat.categories)
    new_order = [reference_class] + [c for c in categories if c != reference_class]
    y_train = y_train.cat.reorder_categories(new_order, ordered=False)
    X_train_const = sm.add_constant(X_train)
    model = sm.MNLogit(y_train, X_train_const)
    result = model.fit(method='lbfgs', maxiter=500) # (method='newton', maxiter=200, disp=False)
    inference_df, article_table = MNL_results(result, other_classes)

    # Confusion matrix
    X_const = sm.add_constant(X_train)
    pred_probs = result.predict(X_const)
    y_pred_idx = pred_probs.idxmax(axis=1)
    mapping = dict(zip(pred_probs.columns, labels))
    y_pred = y_pred_idx.map(mapping)
    disp = ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize="true", cmap="Greens", values_format=".2f")
    disp.ax_.grid(False)
    plt.title("MNL Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path_confusion_matrix, dpi=320, bbox_inches="tight")
    plt.show()
    return inference_df, article_table

