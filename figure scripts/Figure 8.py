# %%
import pickle

import shap

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = "results_pre_perfold_metrics_20250127/lgbm_('epts', 'mri')_sum_epts_calibrated_includeclinical"

model = pickle.load(open(f"{path}/best_model.pkl", "rb"))

X = pd.read_csv(f"{path}/X_final.csv")
y = pd.read_csv(f"{path}/y_final.csv")

# %%

pipeline = model.estimator_.calibrated_classifiers_[0].estimator

explainer = shap.TreeExplainer(pipeline[-1])
transformed_X = pipeline[:-1].transform(X)
feature_names = pipeline[:-1].get_feature_names_out()
shap_values = explainer(transformed_X)

# %%


def extract_values(shap_values):
    return pd.DataFrame(
        (
            zip(
                feature_names[np.argsort(np.abs(shap_values).std(0))],
                np.abs(shap_values).std(0),
            )
        ),
        columns=["feature", "importance"],
    ).sort_values(by="importance", ascending=False)


extract_values(explainer.shap_values(transformed_X))

# %%
plt.figure(figsize=(13, 6))
shap.summary_plot(
    shap_values, transformed_X, feature_names=feature_names, plot_size=None
)
# %%
plt.figure(figsize=(13, 6))
shap.summary_plot(
    shap_values,
    transformed_X,
    feature_names=feature_names,
    plot_size=None,
    plot_type="bar",
)

# %%
aggs = np.abs(explainer.shap_values(transformed_X)).mean(0)
# %%
sv_df = pd.DataFrame(aggs.T)
sv_df = sv_df.sum(1).sort_values(ascending=False)
sv_df.index = feature_names[sv_df.index]
sv_df
# %%
