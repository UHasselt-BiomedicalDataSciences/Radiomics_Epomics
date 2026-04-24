# %%

import argparse
import os
import pickle
import json
from collections import defaultdict
import warnings
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import numpy as np
import pandas as pd
from boruta import BorutaPy
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import ADASYN
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    TunedThresholdClassifierCV,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from utils import drop_correlated_features, load_df


BASE_PATH = "results"
CALIBRATE = False
SEED = 42
POOL_SIZE = 1

BEST_HPARAMS = {}


def mri_columns_transformer(df):
    return df.filter(like="mri_").select_dtypes(include=["float"]).columns


def epts_columns_transformer(df):
    return df.filter(like="epts_").select_dtypes(include=["float"]).columns


def preprocessing_pipeline(df):
    return ColumnTransformer(
        [
            (
                "num_epts",
                make_pipeline(
                    StandardScaler(),
                ),
                epts_columns_transformer(df),
            ),
            (
                "num_mri",
                make_pipeline(
                    StandardScaler(),
                ),
                mri_columns_transformer(df),
            ),
            (
                "cat",
                "passthrough",
                df.select_dtypes(include=["category", "int"]).columns,
            ),
        ]
    )


def make_model(df, model_type):
    if model_type == "logistic":
        return make_pipeline(
            preprocessing_pipeline(df),
            LogisticRegression(
                C=1e-2,
                random_state=SEED,
                class_weight="balanced",
                tol=1e-4,
                max_iter=10000,
            ),
        )
    elif model_type == "random_forest":
        return make_pipeline(
            preprocessing_pipeline(df),
            RandomForestClassifier(
                n_estimators=1000, random_state=SEED, class_weight="balanced", n_jobs=1
            ),
        )
    elif model_type == "balanced_random_forest":
        return make_pipeline(
            preprocessing_pipeline(df),
            BalancedRandomForestClassifier(
                n_estimators=1000,
                random_state=SEED,
                n_jobs=1,
                sampling_strategy="all",
                replacement=True,
            ),
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            preprocessing_pipeline(df),
            LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1),
        )
    else:
        raise ValueError(f"Model type {model_type} not allowed")


def hparam_search_for_model(model_type, model):
    scorer = "average_precision"
    # Better scorer
    # scorer = 'roc_auc'
    if model_type == "logistic":
        return GridSearchCV(
            model,
            param_grid={
                "logisticregression__C": np.logspace(-4, 1, 16),
                "logisticregression__penalty": ["elasticnet"],
                "logisticregression__solver": ["saga"],
                "logisticregression__l1_ratio": np.linspace(0, 1, 15),
                # 'columntransformer__num_mri__pca__n_components': np.arange(1, 7),
            },
            cv=5,
            n_jobs=-1,
            scoring=scorer,
            error_score="raise",
            verbose=0,
        )
    elif model_type == "random_forest":
        return RandomizedSearchCV(
            model,
            param_distributions={
                "randomforestclassifier__n_estimators": randint(10, 1000),
                "randomforestclassifier__max_depth": randint(1, 10),
                "randomforestclassifier__max_features": ["sqrt", "log2"],
                "randomforestclassifier__class_weight": [
                    "balanced",
                    "balanced_subsample",
                ],
                "randomforestclassifier__min_samples_split": randint(2, 10),
                "randomforestclassifier__min_samples_leaf": randint(1, 10),
                "randomforestclassifier__bootstrap": [True, False],
                # 'columntransformer__num_mri__pca__n_components': randint(3, 15),
            },
            cv=5,
            n_jobs=-1,
            random_state=SEED,
            scoring=scorer,
            n_iter=500,
            error_score="raise",
            verbose=0,
        )
    elif model_type == "balanced_random_forest":
        return RandomizedSearchCV(
            model,
            param_distributions={
                "balancedrandomforestclassifier__n_estimators": randint(10, 1000),
                "balancedrandomforestclassifier__max_depth": randint(1, 10),
                "balancedrandomforestclassifier__max_features": ["sqrt", "log2"],
                "balancedrandomforestclassifier__min_samples_split": randint(2, 10),
                "balancedrandomforestclassifier__min_samples_leaf": randint(1, 10),
                "balancedrandomforestclassifier__bootstrap": [True, False],
                # 'columntransformer__num_mri__pca__n_components': randint(3, 15),
            },
            cv=5,
            n_jobs=-1,
            random_state=SEED,
            scoring=scorer,
            n_iter=500,
            error_score="raise",
            verbose=0,
        )
    elif model_type == "lgbm":
        return RandomizedSearchCV(
            model,
            param_distributions={
                "lgbmclassifier__n_estimators": randint(10, 1000),
                "lgbmclassifier__learning_rate": loguniform(1e-4, 1e0),
                "lgbmclassifier__num_leaves": randint(2, 100),
                "lgbmclassifier__feature_fraction": uniform(0.1, 0.9),
                "lgbmclassifier__bagging_fraction": uniform(0.1, 0.9),
                "lgbmclassifier__max_depth": randint(1, 10),
                "lgbmclassifier__max_bin": randint(2, 255),
                "lgbmclassifier__min_data_in_leaf": randint(1, 100),
                "lgbmclassifier__min_sum_hessian_in_leaf": loguniform(1e-4, 1e2),
                # 'columntransformer__num_mri__pca__n_components': randint(3, 15),
            },
            cv=5,
            n_jobs=-1,
            random_state=SEED,
            scoring=scorer,
            n_iter=500,
            error_score="raise",
            verbose=0,
        )
    else:
        raise ValueError(f"Model type {model_type} not allowed")


def feature_selection(X, y):
    non_mri_columns_mask = ~X.columns.str.contains("mri_")

    # Feature selection
    selector = BorutaPy(
        estimator=RandomForestClassifier(
            n_jobs=1, class_weight="balanced", random_state=SEED, bootstrap=True
        ),
        perc=70,
        max_iter=50,
        n_estimators="auto",
        random_state=SEED,
        verbose=0,
    )
    support = selector.fit(X.values, y.values).support_

    # Always keep non-MRI columns
    support[non_mri_columns_mask] = True

    return support


def train_and_predict_single(model_type, train_idx, test_idx, X, y, X_test, y_test):
    X = X.copy()
    y = y.copy().astype(int)

    X_test = X_test.copy()
    y_test = y_test.copy().astype(int)

    X_train, X_test = X.iloc[train_idx], X_test.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y_test.iloc[test_idx]

    idx_to_keep = X_train.drop_duplicates().index
    X_train = X_train.loc[idx_to_keep]
    y_train = y_train.loc[idx_to_keep]

    # Remove outliers
    zscore_threshold = 6
    mask = (np.abs(stats.zscore(X_train)) < zscore_threshold).all(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Feature selection
    support = feature_selection(X_train, y_train)

    X_train = X_train[X_train.columns[support]]
    X_test = X_test[X_test.columns[support]]

    # Resampler
    resampler = ADASYN(
        random_state=SEED, n_neighbors=NearestNeighbors(n_neighbors=3, n_jobs=1)
    )
    X_train, y_train = resampler.fit_resample(X_train, y_train)

    model = make_model(X_train, model_type)
    model.set_params(**BEST_HPARAMS)

    if CALIBRATE:
        model = CalibratedClassifierCV(
            model,
            ensemble=False,
            cv=3,
            method="sigmoid",
        )

    try:
        model = TunedThresholdClassifierCV(
            model,
            cv=0.15,
            n_jobs=1,
            random_state=SEED,
        )
    except ValueError:
        # This happens when the estimator makes constant predictions
        pass

    model.fit(X_train, y_train)

    test_predictions = model.predict_proba(X_test)
    test_predictions_binary = model.predict(X_test)

    selected_features = X_train.columns

    return test_idx, test_predictions, test_predictions_binary, model, selected_features


def find_best_hyperparameters(model_type, X, y):
    # Surpress UserWarning
    import warnings

    warnings.filterwarnings("ignore")
    model = make_model(X, model_type)
    hparam_search = hparam_search_for_model(model_type, model)

    # If no hyperparameter search is needed, return an empty dict
    if hparam_search is None:
        return {}
    else:
        search_result = hparam_search.fit(X, y)
        return search_result.best_params_


class RepeatedStratifiedGroupKFold(StratifiedGroupKFold):
    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.n_repeats = n_repeats

    def split(self, X, y=None, groups=None):
        for i in range(self.n_repeats):
            super().__init__(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state + i
            )
            yield from super().split(X, y, groups)


def train_and_predict(df, test_df, model_type):
    # Prepare the df, by getting rid of any weird indexing
    df = df.reset_index(drop=True)
    X = df.drop(columns=["clinic_id", "Worsening"])
    y = df["Worsening"]

    # Prepare the test_df
    test_df = test_df.reset_index(drop=True)
    test_groups = test_df["clinic_id"]
    test_X = test_df.drop(columns=["clinic_id", "Worsening"])
    test_y = test_df["Worsening"]

    # Split using RepeatedStratifiedGroupKFold
    splitter = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=20, random_state=SEED)

    # Prepare the data structures to store the predictions
    all_predictions = defaultdict(list)
    metrics = [
        "accuracy",
        "f1",
        "log_loss",
        "geometric_mean",
        "roc_auc",
        "aupr",
        "sensitivity",
        "specificity",
        "brier",
        "mcc",
    ]
    pretty_names = {
        "accuracy": "B.Acc.",
        "f1": "F1",
        "log_loss": "NLL",
        "geometric_mean": "Geo μ",
        "roc_auc": "AUROC",
        "aupr": "AUPR",
        "sensitivity": "Sens",
        "specificity": "Spec",
        "brier": "Brier",
        "mcc": "MCC",
    }
    all_metrics = {m: [] for m in metrics}

    all_curves = defaultdict(list)

    def update_predictions(i, pbar):
        y_true = []
        y_pred = []
        y_pred_binary = []

        for idx, prediction, prediction_binary in zip(i[0], i[1], i[2]):
            all_predictions[idx].append(prediction[1])
            y_true.append(y.iloc[idx])
            y_pred.append(prediction[1])
            y_pred_binary.append(prediction_binary)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = np.array(y_pred_binary)

        # mean_predictions = {k: np.mean(v) for k, v in all_predictions.items()}
        # final_y = df.loc[mean_predictions.keys()]['Worsening']
        # pred_values = [p for p in mean_predictions.values()]

        accuracy = balanced_accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        log_loss_value = log_loss(y_true, y_pred, labels=[0, 1])
        geometric_mean = geometric_mean_score(y_true, y_pred_binary)
        sensitivity = sensitivity_score(y_true, y_pred_binary)
        specificity = specificity_score(y_true, y_pred_binary)
        brier = brier_score_loss(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred_binary)

        if y_true.astype(int).sum() > 0:
            roc_auc = roc_auc_score(y_true, y_pred)
            aupr = average_precision_score(y_true, y_pred)
        else:
            warnings.warn("No positive samples in this fold")
            roc_auc = 0.0
            aupr = 0.0

        all_metrics["accuracy"].append(accuracy)
        all_metrics["f1"].append(f1)
        all_metrics["log_loss"].append(log_loss_value)
        all_metrics["geometric_mean"].append(geometric_mean)
        all_metrics["roc_auc"].append(roc_auc)
        all_metrics["aupr"].append(aupr)
        all_metrics["sensitivity"].append(sensitivity)
        all_metrics["specificity"].append(specificity)
        all_metrics["brier"].append(brier)
        all_metrics["mcc"].append(mcc)

        all_curves["roc"].append(roc_curve(y_true, y_pred))
        all_curves["prc"].append(precision_recall_curve(y_true, y_pred))

        m_values = [all_metrics[m] for m in metrics]

        s = ""

        for i, m in enumerate(metrics):
            s += f"{pretty_names[m]}:{np.mean(m_values[i]):.2f}, "

        pbar.set_description(s)

    all_arguments = [
        (
            model_type,
            df[~df["clinic_id"].isin(pd.unique(test_groups.iloc[test_idx]))].index,
            test_df[
                test_df["clinic_id"].isin(pd.unique(test_groups.iloc[test_idx]))
            ].index,
            X,
            y,
            test_X,
            test_y,
        )
        for train_idx, test_idx in splitter.split(test_X, test_y, test_groups)
    ]

    with tqdm_joblib(
        tqdm(desc="My calculation", total=len(all_arguments))
    ) as progress_bar:
        results = Parallel(n_jobs=POOL_SIZE)(
            delayed(train_and_predict_single)(*arg) for arg in all_arguments
        )
        for i, result in enumerate(results):
            update_predictions(result, progress_bar)
            os.makedirs(f"{BASE_PATH}/folds/{i}", exist_ok=True)
            model = result[3]
            with open(f"{BASE_PATH}/folds/{i}/model.pkl", "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f"{BASE_PATH}/folds/{i}/selected_features.pkl", "wb") as f:
                pickle.dump(result[4], f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f"{BASE_PATH}/folds/{i}/curves.pkl", "wb") as f:
                pickle.dump(all_curves, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print metrics as csv, with mean and std
    for m in metrics:
        print(f"{m},{m}_std", end=",")
    print()
    for m in metrics:
        print(f"{np.mean(all_metrics[m]):.3f},{np.std(all_metrics[m]):.3f}", end=",")
    print()

    # Export those metrics to a file
    with open(f"{BASE_PATH}/metrics.csv", "w") as f:
        for m in metrics:
            f.write(f"{m},{m}_std,")
        f.write("\n")
        for m in metrics:
            f.write(f"{np.mean(all_metrics[m]):.3f},{np.std(all_metrics[m]):.3f},")
        f.write("\n")

    # Export per-fold metrics
    with open(f"{BASE_PATH}/per_fold_metrics.csv", "w") as f:
        for m in metrics:
            f.write(f"{m},")
        f.write("\n")
        for i in range(len(results)):
            for m in metrics:
                f.write(f"{all_metrics[m][i]},")
            f.write("\n")

    mean_predictions = {k: np.mean(v) for k, v in all_predictions.items()}
    final_y = df.loc[mean_predictions.keys()]["Worsening"]
    mean_predictions = list(mean_predictions.values())
    return mean_predictions, final_y


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# %%


ALLOWED_MODEL_TYPES = [
    "logistic",
    "random_forest",
    "lgbm",
    "balanced_random_forest",
]


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_types", nargs="+", default=["epts"])
    parser.add_argument("--model_type", type=str, default="logistic")
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--sum_epts", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--include_clinical", action="store_true")
    parser.add_argument("--permutation_test", action="store_true")
    args = parser.parse_args()

    CALIBRATE = args.calibrate
    model_type = args.model_type
    assert model_type in ALLOWED_MODEL_TYPES, f"Model type {model_type} not allowed"

    data_types = tuple(args.data_types[0].split(","))

    _, test_df = load_df(data_types, delta=args.delta, sum_epts=args.sum_epts)

    rng = np.random.default_rng(SEED)

    def process_df(df):
        if "ts" in df.columns:
            df = df.drop(columns=["ts"])

        # Make all columns that only have a few unique values categorical
        for column in df.columns:
            if df[column].nunique() < 10:
                df[column] = df[column].astype("category")
            else:
                df[column] = df[column].astype("float64")

        df.clinic_id = df.clinic_id.astype("int")

        # Onehot encode categorical columns
        df = pd.get_dummies(df, dtype=int)

        if "Worsening_0.0" in df.columns:
            df["Worsening"] = df["Worsening_1.0"].astype("category")
            df.drop(columns=["Worsening_0.0", "Worsening_1.0"], inplace=True)
        if "Worsening_False" in df.columns:
            df["Worsening"] = df["Worsening_True"].astype("category")
            df.drop(columns=["Worsening_False", "Worsening_True"], inplace=True)

        if args.permutation_test:
            # Permute the Worsening column
            df["Worsening"] = rng.permutation(df["Worsening"].values)
        return df

    if not args.include_clinical:
        # Drop all columns that start with clinical_
        test_df = test_df.drop(columns=test_df.filter(regex="^clinical_").columns)
        assert "clinical_age" not in test_df.columns, "clinical_age in columns"
    else:
        assert "clinical_age" in test_df.columns, "clinical_age not in columns"
        assert "clinical_Gender" in test_df.columns, "clinical_Gender not in columns"

    test_df = drop_correlated_features(test_df, threshold=0.90)
    test_df = process_df(test_df)

    df_for_init = test_df.reset_index(drop=True)

    X_for_hparams = df_for_init.drop(columns=["Worsening", "clinic_id"])
    y_for_hparams = df_for_init["Worsening"].astype(int)
    groups_for_hparams = df_for_init["clinic_id"]

    cols_for_hparams = feature_selection(X_for_hparams, y_for_hparams)
    X_for_hparams = X_for_hparams[X_for_hparams.columns[cols_for_hparams]]
    # resampler = ADASYN(random_state=SEED, n_neighbors=3)
    # X_for_hparams_resamp, y_for_hparams_resamp = resampler.fit_resample(X_for_hparams, y_for_hparams)
    X_for_hparams_resamp, y_for_hparams_resamp = X_for_hparams, y_for_hparams

    BEST_HPARAMS = find_best_hyperparameters(
        model_type, X_for_hparams_resamp, y_for_hparams_resamp
    )

    # Build base_path
    BASE_PATH = f"results/{model_type}"

    BASE_PATH += f"_{str(data_types)}"
    if args.sum_epts:
        BASE_PATH += "_sum_epts"
    if args.delta:
        BASE_PATH += "_delta"
    if args.calibrate:
        BASE_PATH += "_calibrated"
    if args.include_clinical:
        BASE_PATH += "_includeclinical"

    os.makedirs(f"{BASE_PATH}", exist_ok=True)
    with open(f"{BASE_PATH}/hparams.json", "w") as f:
        json.dump(BEST_HPARAMS, f, indent=4, cls=NpEncoder)

    # Do the evaluation on all folds
    mean_predictions, y = train_and_predict(test_df, test_df, model_type=model_type)

    np.save(f"{BASE_PATH}/mean_predictions.npy", mean_predictions)
    np.save(f"{BASE_PATH}/y.npy", y)

    # Save the "best" model
    res = train_and_predict_single(
        model_type,
        df_for_init.index,
        df_for_init.index,
        X_for_hparams,
        y_for_hparams,
        X_for_hparams,
        y_for_hparams,
    )

    test_idx, test_predictions, test_predictions_binary, model, selected_features = res

    with open(f"{BASE_PATH}/best_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{BASE_PATH}/best_selected_features.pkl", "wb") as f:
        pickle.dump(selected_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    X_for_hparams.to_csv(f"{BASE_PATH}/X_final.csv", index=False)
    y_for_hparams.to_csv(f"{BASE_PATH}/y_final.csv", index=False)
    groups_for_hparams.to_csv(f"{BASE_PATH}/groups_final.csv", index=False)

    np.save(f"{BASE_PATH}/best_predictions.npy", test_predictions)
