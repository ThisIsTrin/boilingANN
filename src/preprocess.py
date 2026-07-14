import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from typing import Tuple


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df["delta_T_bin"] = pd.qcut(df["delta_T"], q=5, labels=False, duplicates="drop")

    df["group_id"] = (
        df["phi"].astype(str) + "_" + df["material"] + "_" + df["ra"].astype(str)
    )

    unique_groups = df["group_id"].unique()
    group_bins = df.groupby("group_id")["delta_T_bin"].agg(lambda x: x.mode()[0])

    train_groups, val_groups = train_test_split(
        unique_groups, test_size=0.2, stratify=group_bins, random_state=67
    )

    train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["group_id"].isin(val_groups)].reset_index(drop=True)

    return train_df, val_df


def main():
    os.makedirs("data", exist_ok=True)

    # Header: phi|gamma|ra|q_vol|q_Wm2|q_Wm2|material|seed|log_q|log_ra|r_d|C1|C3|bubble_vol_factor|q_star|delta_T|log_delta_T|HTC|log_HTC|T_wall_ss|T_wall_std|n_sites_ss|q_out_ss|q_mc_ss|q_me_ss|q_nc_ss|q_rad_ss|br_ss|sim_name
    df = pd.read_csv("data/raw/simulation_dataset.csv")
    df.head()

    # FEATURES = ["phi", "gamma", "log_q", "log_ra", "r_d", "C1", "bubble_vol_factor"]
    # FEATURES = ["phi", "log_q", "log_ra", "r_d", "C1", "bubble_vol_factor"]
    FEATURES = [
        "phi",
        "k_s",
        "rho_s",
        "c_ps",
        "log_q",
        "log_ra",
        "r_d",
        "C1",
        "bubble_vol_factor",
    ]
    TARGET = "log_delta_T"
    TARGET_RAW = "delta_T"

    # if (df.isnull().sum()) > 0:
    #    print("WARNING - Null detected")

    df = df[(df["delta_T"] > 0) & (df["delta_T"] < 150)]

    df = df[np.isfinite(df[FEATURES + [TARGET]].values).all(axis=1)].reset_index(
        drop=True
    )

    df = df.dropna()

    train_df, val_df = split_data(df)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train = scaler_x.fit_transform(train_df[FEATURES])
    y_train = scaler_y.fit_transform(train_df[[TARGET]]).ravel()

    x_val = scaler_x.transform(val_df[FEATURES])
    y_val = scaler_y.transform(val_df[[TARGET]]).ravel()

    y_train_raw = train_df[TARGET_RAW].values
    y_val_raw = val_df[TARGET_RAW].values

    np.save("data/x_train.npy", x_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/x_val.npy", x_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/y_train_raw.npy", y_train_raw)
    np.save("data/y_val_raw.npy", y_val_raw)

    with open("models/scaler_x.pkl", "wb") as f:
        pickle.dump(scaler_x, f)
    with open("models/scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    with open("models/feature_names.json", "w") as f:
        json.dump(FEATURES, f)


if __name__ == "__main__":
    main()
