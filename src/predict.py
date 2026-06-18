import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
import os
from model import BoilingANN
from dataclasses import dataclass, asdict
from pathlib import Path
from matplotlib import pyplot as plt


@dataclass
class PredictionConfig:

    # Output
    output_dir: str = "outputs/predictions"
    output_filename: str = "jiang2021_ann.csv"

    # CSV
    exp_csv: str = "data/experiment/jiang2021_ex.csv"
    sim_csv: str = "data/automata/jiang2021_sim_benjamin.csv"

    # Physical constants
    sigma: float = 0.0589  # Surface tension [N/m]
    g: float = 9.81  # Gravity [m/s^2]
    rho_l: float = 958.0  # Liquid density [kg/m^3]
    rho_v: float = 0.597  # Vapor density [kg/m^3]

    # Surface parameters
    phi_deg: float = 78.6  # Contact angle [degrees]
    gamma: float = 22.21  # Roughness factor [dimensionless]
    ra: float = 0.05e-6  # Roughness [m]

    # Heat flux range
    q_min_wcm2: float = 1.0  # Min heat flux [W/cm^2]
    q_max_wcm2: float = 150.0  # Max heat flux [W/cm^2]
    n_points: int = 250  # Number of points in curve

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def load_model(num_features):
    try:
        model = BoilingANN(num_features)
        model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        raise RuntimeError("Model not found: models/best_model.pt")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def load_utils():
    try:
        with open("models/scaler_x.pkl", "rb") as f:
            scaler_x = pickle.load(f)

        with open("models/scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)

        with open("models/feature_names.json") as f:
            feature_names = json.load(f)

        return scaler_x, scaler_y, feature_names
    except FileNotFoundError as e:
        raise RuntimeError(f"File not found: {e}")


def geometric_feature(phi_deg, sigma, g, rho_l, rho_v):
    phi_rad = np.radians(phi_deg)
    r_d = 0.5 * 0.0148 * phi_deg * np.sqrt(2 * sigma / (g * (rho_l - rho_v)))

    # Contact angle factor
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    C1 = (1 + cos_phi) / (sin_phi + 1e-8)

    # Bubble volume factor (corrected from geometry)
    bubble_vol_factor = 4 - (2 + cos_phi) * (1 - cos_phi) ** 2

    return r_d, C1, bubble_vol_factor


def feature_dataframe(
    q_Wcm2_array, features, phi_deg, gamma, ra, r_d, C1, bubble_vol_factor
):

    df = pd.DataFrame(
        {
            "phi": phi_deg,
            "gamma": gamma,
            "log_q": np.log10(q_Wcm2_array),
            "log_ra": np.log10(ra),
            "r_d": r_d,
            "C1": C1,
            "bubble_vol_factor": bubble_vol_factor,
        }
    )

    return df[features]


def predict(feature_df, model, scaler_x, scaler_y):
    # Scale
    x_scaled = scaler_x.transform(feature_df.values)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Gen predictions
    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy()

    # Inverse scale and log scale
    log_dT = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    dT = 10**log_dT

    return dT


def save(dT, q_Wcm2_array, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({0: dT, 1: q_Wcm2_array})
    if os.path.exists(output_path):
        os.remove(output_path)

    out.to_csv(output_path, index=False, header=False)


def clean_for_plot(dT, q):
    dT = np.asarray(dT, dtype=float)
    q = np.asarray(q, dtype=float)
    m = (dT > 0) & (q > 0) & np.isfinite(dT) & np.isfinite(q)
    dT2, q2 = dT[m], q[m]
    idx = np.argsort(dT2)
    return dT2[idx], q2[idx]


def plot(exp_csv, sim_csv, output_path, output_dir):
    df_exp = pd.read_csv(exp_csv, header=None, usecols=[0, 1], names=["dT", "q"])
    dT_exp_p, q_exp_p = clean_for_plot(df_exp["dT"].values, df_exp["q"].values)

    df_sim = pd.read_csv(sim_csv, header=None)
    dT_rep = df_sim.iloc[:, 0:10].to_numpy(dtype=float)
    q_sim_raw = df_sim.iloc[:, 10].to_numpy(dtype=float)
    dT_sim_p, q_sim_p = clean_for_plot(np.mean(dT_rep, axis=1), q_sim_raw)

    df_ann = pd.read_csv(output_path, header=None, usecols=[0, 1], names=["dT", "q"])
    dT_ann_p, q_ann_p = clean_for_plot(df_ann["dT"].values, df_ann["q"].values)

    COLORS = {
        "exp": "#1a1a2e",
        "sim": "#e07b39",
        "ann": "#2e7d59",
    }

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")

    # Experiment
    ax.scatter(
        dT_exp_p,
        q_exp_p,
        color=COLORS["exp"],
        s=28,
        zorder=4,
        marker="o",
        linewidths=0.4,
        edgecolors="white",
        label="Experiment (Jiang 2021)",
    )

    # Automata simulation
    ax.scatter(
        dT_sim_p,
        q_sim_p,
        color=COLORS["sim"],
        s=28,
        zorder=3,
        marker="s",
        linewidths=0.4,
        edgecolors="white",
        label="Automata simulation",
    )

    # ANN prediction — line + light fill for confidence feel
    ax.plot(
        dT_ann_p,
        q_ann_p,
        color=COLORS["ann"],
        linewidth=2.0,
        zorder=5,
        label="ANN",
        solid_capstyle="round",
    )
    ax.fill_between(
        dT_ann_p,
        q_ann_p * 0.85,
        q_ann_p * 1.15,
        color=COLORS["ann"],
        alpha=0.10,
        zorder=2,
        label="ANN ±15%",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Wall superheat, $\Delta T$ (K)", labelpad=7)
    ax.set_ylabel(r"Heat flux, $q''$ (W cm$^{-2}$)", labelpad=7)

    ax.tick_params(which="both", top=True, right=True)

    ax.legend(
        loc="upper left",
    )

    plt.savefig(
        Path(output_dir) / "loglog.jpg",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


def generate_boiling_curve(config: PredictionConfig):
    """
    Pipeline:
    1. Load model and scalers
    2. Validate configuration
    3. Calculate geometric features
    4. Build feature dataframe
    5. Generate predictions
    6. Save results
    """

    scaler_x, scaler_y, feature_names = load_utils()

    model = load_model(len(feature_names))

    r_d, C1, bubble_vol_factor = geometric_feature(
        config.phi_deg, config.sigma, config.g, config.rho_l, config.rho_v
    )

    q_Wcm2_array = np.logspace(
        np.log10(config.q_min_wcm2), np.log10(config.q_max_wcm2), config.n_points
    )

    feature_df = feature_dataframe(
        q_Wcm2_array,
        feature_names,
        config.phi_deg,
        config.gamma,
        config.ra,
        r_d,
        C1,
        bubble_vol_factor,
    )

    dT = predict(feature_df, model, scaler_x, scaler_y)

    output_path = Path(config.output_dir) / config.output_filename
    save(dT, q_Wcm2_array, output_path)

    plot(config.exp_csv, config.sim_csv, output_path, config.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate boiling curves using trained ANN model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = PredictionConfig.from_dict(config_dict)
    else:
        config = PredictionConfig()

    generate_boiling_curve(config)
