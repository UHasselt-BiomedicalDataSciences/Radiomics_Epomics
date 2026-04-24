import pandas as pd
import numpy as np
from warnings import simplefilter
import os

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_PATH = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "PROCESSED_DATA", "processed_datasets_20251216")
)
BASE_PATH = os.path.normpath(os.environ.get("MS_DATA_BASE_PATH", DEFAULT_BASE_PATH))


def apply_sum_epts(df):
    for column in df.columns:
        if column.startswith("epts_") and column.endswith("_r"):
            column_l = column[:-2] + "_l"

            df.rename(
                columns={
                    column: "r_" + column[:-2],
                    column_l: "l_" + column[:-2],
                },
                inplace=True,
            )

    for column in df.columns:
        if column.startswith("r_"):
            column_l = "l_" + column[2:]
            if column_l in df.columns:
                df[column[2:]] = df[column] + df[column_l]
                df.drop(columns=[column, column_l], inplace=True)

    return df


ONE_YEAR_IN_S = 365.256366 * 24 * 60 * 60
HALF_YEAR_IN_S = ONE_YEAR_IN_S / 2
QUARTER_YEAR_IN_S = ONE_YEAR_IN_S / 4


def worsening(ed0, ed1):
    """
    Calculates disability progression between EDSS at two different timepoints

    Parameters:
    ed0 (float): EDSS score at the first timepoint
    ed1 (float): EDSS score at the second timepoint

    Returns:
    bool: True if there is disability progression, False otherwise
    """

    if ed0 == 0:
        threshold = 0.5
    elif ed0 <= 5.5:
        threshold = 1 + ed0
    elif ed0 > 5.5:
        threshold = 0.5 + ed0

    return ed1 >= threshold


def make_label_df():
    # Cache the label df
    if os.path.exists(f"{BASE_PATH}/label_df.csv"):
        label_df = pd.read_csv(f"{BASE_PATH}/label_df.csv", index_col=0)
        return label_df

    label_df = pd.read_csv(f"{BASE_PATH}/clinical.csv", index_col=0)
    label_df = label_df[["clinic_id", "edss", "ts", "age", "Gender", "mscourse", "DMT"]]

    for i, row in label_df.iterrows():
        clinic_id = row.clinic_id
        ts = row.ts
        edss = row.edss

        lower_limit = ts + 2 * ONE_YEAR_IN_S
        upper_limit = ts + 3 * ONE_YEAR_IN_S

        mask = label_df.clinic_id == clinic_id
        mask &= label_df.ts >= lower_limit
        mask &= label_df.ts <= upper_limit

        if mask.any():
            next_row = label_df[mask].sort_values("ts").iloc[0]
            next_edss = next_row.edss
            label_df.loc[i, "disability_progression"] = worsening(edss, next_edss)

    label_df = label_df.dropna()

    label_df.to_csv(f"{BASE_PATH}/label_df.csv")

    return label_df


def find_closest_labels(df, label_df):
    for clinic_id, grp in df.groupby("clinic_id"):
        label_grp = label_df[label_df.clinic_id == clinic_id]

        # If the clinic_id is not in the label dataset, skip
        if label_grp.empty:
            continue

        # Go over each row in the group
        for i, row in grp.iterrows():
            # Take the timestamp of the row
            ts = row.ts

            lower_limit = ts - HALF_YEAR_IN_S
            upper_limit = ts + QUARTER_YEAR_IN_S

            mask = label_grp.ts >= lower_limit
            mask &= label_grp.ts <= upper_limit

            if mask.any():
                # Find the closest timestamp of the label dataset wrt the row's timestamp
                indices = (label_grp.ts - ts).abs().argsort()
                # Take the closest timestamp's iloc index
                closest_ts_iloc_idx = indices.iloc[0]
                # Take the row with the closest timestamp
                closest_row = label_grp.iloc[closest_ts_iloc_idx]
                # Take the index of the closest row
                label_ts_idx = closest_row.name

                label_row = label_df.loc[label_ts_idx]
                df.loc[i, "Worsening"] = label_row.disability_progression == 1
                df.loc[i, "clinical_edss"] = label_row.edss
                df.loc[i, "clinical_age"] = label_row.age
                df.loc[i, "clinical_Gender"] = (label_row.Gender == "F") * 1
                # df.loc[i, 'clinical_mscourse'] = label_row.mscourse

    df = df.dropna(subset=["Worsening"])

    return df


def load_single_df_mri(
    delta=False, remove_outliers=False, keep_ts=False, own_labels=True
):
    datatype = "mri"
    df = pd.read_csv(f"{BASE_PATH}/{datatype}.csv", index_col=0)
    df.drop(
        columns=[
            c
            for c in df.columns
            if "MRIpipeline" in c or "mri_disability_progression" in c
        ],
        inplace=True,
    )

    if own_labels:
        label_df = make_label_df()
        df = find_closest_labels(df, label_df)
        df.drop(columns=["disability_progression"], inplace=True)
    else:
        df.rename(columns={"disability_progression": "Worsening"}, inplace=True)
        df["Worsening"] = df["Worsening"].astype(bool)

    if delta:
        # Create _0 and _1 columns for each feature, where the _1 column is the closest timepoint to
        # 6 months after the _0 column
        copy_df = df.copy()
        for i, row in copy_df.iterrows():
            t1_ts = row.ts + HALF_YEAR_IN_S
            lower_limit = t1_ts - QUARTER_YEAR_IN_S
            upper_limit = t1_ts + QUARTER_YEAR_IN_S

            mask = copy_df.clinic_id == row.clinic_id
            mask &= copy_df.ts >= lower_limit
            mask &= copy_df.ts <= upper_limit

            pat_grp = copy_df[mask]

            if pat_grp.empty:
                continue

            indices = (pat_grp.ts - t1_ts).abs().argsort()
            closest_ts_iloc_idx = indices.iloc[0]
            closest_row = pat_grp.iloc[closest_ts_iloc_idx]
            closest_ts = closest_row.ts
            closest_idx = closest_row.name

            for column in copy_df.columns:
                if column.startswith("mri_"):
                    column_1 = column + "_1"
                    assert closest_idx != i
                    df.loc[i, column_1] = df.loc[closest_idx, column].copy()

        for column in df.columns:
            if not column in ["ts", "Worsening", "clinic_id"] and not column.endswith(
                "_1"
            ):
                df.rename(columns={column: column + "_0"}, inplace=True)

        df = df.dropna()

    # Rename all columns to mri_{column_name}
    df.rename(
        columns={
            c: f"mri_{c}"
            for c in df.columns
            if c
            not in [
                "clinic_id",
                "ts",
                "Worsening",
                "clinical_edss",
                "clinical_age",
                "clinical_Gender",
            ]
        },
        inplace=True,
    )

    return df


def load_reference_df(delta=False, sum_epts=False):
    epts_df = load_single_df_epts(delta=delta, sum_epts=sum_epts, keep_ts=True)
    mri_df = load_single_df_mri(delta, keep_ts=True)

    mri_df.drop(columns=["Worsening"], inplace=True)

    # Match each MRI scan with the closest EPTS scan
    for i, row in mri_df.iterrows():
        clinic_id = row.clinic_id
        ts = row.ts

        lower_limit = ts - HALF_YEAR_IN_S
        upper_limit = ts + QUARTER_YEAR_IN_S

        mask = epts_df.clinic_id == clinic_id
        mask &= epts_df.ts >= lower_limit
        mask &= epts_df.ts <= upper_limit

        if mask.any():
            epts_row = epts_df[mask].sort_values("ts").iloc[0]
            epts_row = epts_row.drop(["clinic_id", "ts"])

            mri_df.loc[i, epts_row.index] = epts_row

    final_df = mri_df.dropna()

    return final_df


def load_single_df_epts(
    delta=False, sum_epts=False, remove_outliers=False, keep_ts=False
):
    datatype = "epts"
    df = pd.read_csv(f"{BASE_PATH}/{datatype}.csv", index_col=0)

    df = df.drop(columns=["date"])

    # Rename all non-clinic_id, ts or Worsening columns to epts_{column_name}
    df.rename(
        columns={
            c: f"epts_{c}"
            for c in df.columns
            if c not in ["clinic_id", "ts", "Worsening"]
        },
        inplace=True,
    )

    if datatype == "mri":
        raise NotImplementedError
        df["Worsening"] = df["disability_progression"].astype(float)
        df.drop(columns=["disability_progression"], inplace=True)
        df.drop(
            columns=[
                c
                for c in df.columns
                if "MRIpipeline" in c or "mri_disability_progression" in c
            ],
            inplace=True,
        )

    if sum_epts:
        df = apply_sum_epts(df)

    if delta:
        # Create _0 and _1 columns for each feature, where the _1 column is the closest timepoint to
        # 6 months after the _0 column
        copy_df = df.copy()
        for i, row in copy_df.iterrows():
            t1_ts = row.ts + HALF_YEAR_IN_S
            lower_limit = t1_ts - QUARTER_YEAR_IN_S
            upper_limit = t1_ts + QUARTER_YEAR_IN_S

            mask = copy_df.clinic_id == row.clinic_id
            mask &= copy_df.ts >= lower_limit
            mask &= copy_df.ts <= upper_limit

            pat_grp = copy_df[mask]

            if pat_grp.empty:
                continue

            indices = (pat_grp.ts - t1_ts).abs().argsort()
            closest_ts_iloc_idx = indices.iloc[0]
            closest_row = pat_grp.iloc[closest_ts_iloc_idx]
            closest_ts = closest_row.ts
            closest_idx = closest_row.name

            for column in copy_df.columns:
                if column.startswith("epts_"):
                    column_1 = column + "_1"
                    assert closest_idx != i
                    df.loc[i, column_1] = df.loc[closest_idx, column].copy()

        for column in df.columns:
            if not column in ["ts", "Worsening", "clinic_id"] and not column.endswith(
                "_1"
            ):
                df.rename(columns={column: column + "_0"}, inplace=True)

        df = df.dropna()

    label_df = make_label_df()

    df = find_closest_labels(df, label_df)

    # If delta, calculate the relative change in each feature, where t0 and t1 end with _0 and _1
    for column in df.columns:
        if column.endswith("_0"):
            column_1 = column[:-1] + "1"
            if column_1 in df.columns:
                if delta:
                    df[column[:-2]] = (df[column_1] - df[column]) / np.abs(df[column])
                else:
                    df[column[:-2]] = df[column]
                df.drop(columns=[column, column_1], inplace=True)

    if remove_outliers:
        raise NotImplementedError

    if not keep_ts:
        df = df.drop(columns=["ts"])

    test_df = df[df.clinic_id.isin(label_df.clinic_id.unique())].copy()

    # Put columns in same order, and assert they are the same
    assert len(df.columns) == len(test_df.columns)
    df = df[test_df.columns]
    assert (df.columns == test_df.columns).all()

    return df  # , test_df


def load_df(datatypes, delta=False, sum_epts=False, remove_outliers=False):
    """
    Load a DataFrame from a CSV file and perform data preprocessing.

    Parameters:
    - datatypes (str or list of str): The column names to include in the DataFrame.

    Returns:
    - df (pandas.DataFrame): The loaded and preprocessed DataFrame.
    """

    if isinstance(datatypes, str):
        datatypes = [datatypes]

    ref_df = load_reference_df(delta=delta, sum_epts=sum_epts)

    if len(datatypes) == 1 and datatypes[0] == "epts":
        epts_df = load_single_df_epts(delta=delta, sum_epts=sum_epts)

        ref_df = ref_df[epts_df.columns]

        return epts_df, ref_df
    elif len(datatypes) == 1 and datatypes[0] == "mri":
        mri_df = load_single_df_mri(delta=delta)
        ref_df = ref_df[mri_df.columns]

        return mri_df, ref_df
    elif len(datatypes) == 1 and datatypes[0] == "clinical":
        ref_df = ref_df[
            [
                c
                for c in ref_df.columns
                if c.startswith("clinical_") or c in ["clinic_id", "ts", "Worsening"]
            ]
        ]

        return ref_df, ref_df

    elif len(datatypes) == 2:
        epts_df = load_single_df_epts(delta=delta, sum_epts=sum_epts)
        mri_df = load_single_df_mri(delta=delta)
        cols = epts_df.columns.tolist() + mri_df.columns.tolist()
        # Remove duplicates
        cols = list(dict.fromkeys(cols))
        ref_df = ref_df[cols]

        return epts_df, ref_df


def drop_correlated_features(df, threshold=0.85):
    """
    Drops highly correlated features, keeping one representative feature from each group.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the features.
    threshold (float): The correlation threshold above which features are considered highly correlated.

    Returns:
    pd.DataFrame: A DataFrame with reduced features.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identify columns to drop
    to_drop = set()
    for column in upper_triangle.columns:
        # Check if any feature is highly correlated with the current column
        correlated_features = upper_triangle[column][
            upper_triangle[column] > threshold
        ].index.tolist()

        # If there are correlated features, drop all except one (the current column)
        if correlated_features:
            to_drop.update(correlated_features)

    # Keep only unique columns to drop
    to_drop = list(to_drop)

    # Drop the correlated features from the DataFrame
    reduced_df = df.drop(columns=to_drop)

    # print(f"Features removed: {len(to_drop)}")
    # print(f"Remaining features: {reduced_df.shape[1]}")

    return reduced_df
