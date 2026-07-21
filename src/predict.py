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

    name: str

    # Output
    output_dir: str
    output_filename: str

    # CSV
    # Experimental CSV: 2 columns -> [dT, q]
    # Simulation CSVs: 11 columns -> [dT_1 ... dT_10, q]
    exp_csv: str
    sim_csv: str

    k_s: float  # [W/mK]
    rho_s: float  # [kg/m^3]
    c_ps: float  # [J/kgK]

    # Surface parameters
    phi_deg: float  # Contact angle [degrees]
    # gamma: float # Roughness factor [dimensionless]
    ra: float  # Roughness [m]

    # Heat flux range
    q_min_wcm2: float  # Min heat flux [W/cm^2]
    q_max_wcm2: float  # Max heat flux [W/cm^2]
    n_points: int  # Number of points in curve

    # Physical constants
    k_l: float = 0.679
    rho_l: float = 958.4  # Liquid density [kg/m^3]
    c_pl: float = 4216.0  # [J/kgK]
    sigma: float = 0.0589  # Surface tension [N/m]
    g: float = 9.81  # Gravity [m/s^2]
    rho_v: float = 0.597  # Vapor density [kg/m^3]
    kl: float = 0.679  # thermal conductivity of liquid [W/mK]
    hlv: float = 2.257e6  # latent heat of vaporization [J/kg]
    mu_l: float = 2.82e-4  # dynamic viscosity of liquid [Pa*s]

    @property
    def gamma(self):
        return np.sqrt(
            (self.k_s * self.rho_s * self.c_ps) / (self.k_l * self.rho_l * self.c_pl)
        )

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
        with open("models/feature_names.json") as f:
            feature_names = json.load(f)

        with open("models/scaler_x.pkl", "rb") as f:
            scaler_x = pickle.load(f)

        with open("models/scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)

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


def htc(q_Wcm2, dT):
    # q / (A * (t_surface - t_bulk))
    return q_Wcm2 / np.asarray(dT)


def liCorreleation(dT_array, config):
    theta_deg = max(config.phi_deg, 15.0)
    theta_rad = np.radians(theta_deg)

    ra_um = config.ra * 1e6

    ca_term = (1 - np.cos(theta_rad)) ** 0.5
    ra_term = 1 + 5.45 / ((ra_um - 3.5) ** 2 + 2.61)
    gamma_term = config.gamma ** (-0.04)

    Cs = ca_term * ra_term * gamma_term

    inv_capillary_length = np.sqrt(
        config.g * (config.rho_l - config.rho_v) / config.sigma
    )

    q_w = (
        518503
        * Cs
        * config.kl**3.03
        / (config.hlv * config.mu_l) ** 2.03
        * inv_capillary_length
        * dT_array**3.03
    )

    return q_w


def feature_dataframe(
    q_Wcm2_array,
    features,
    phi_deg,
    k_s,
    rho_s,
    c_ps,
    ra,
    r_d,
    C1,
    bubble_vol_factor,
):

    df = pd.DataFrame(
        {
            "phi": phi_deg,
            "k_s": k_s,
            "rho_s": rho_s,
            "c_ps": c_ps,
            "log_q": np.log10(q_Wcm2_array),
            # "q_Wcm2": q_Wcm2_array,
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


def compute_rmse(dT_ann, q_ann, dT_exp, q_exp):
    """
    RMSE between ANN prediction and experimental data,
    matching Eq. (31) in Kim & Kim (2020): interpolate the ANN curve
    onto the experimental q points, then compare in heat-flux space.
    """
    # interpolate ANN curve (sorted by dT) onto experimental dT values
    q_ann_interp = np.interp(dT_exp, dT_ann, q_ann)

    rmse = np.sqrt(np.mean((q_exp - q_ann_interp) ** 2))
    return rmse


def compute_percent_error(dT_ann, q_ann, dT_exp, q_exp):
    """
    Mean absolute percentage error (MAPE) between ANN prediction
    and experimental data, interpolated onto experimental dT points.
    """
    q_ann_interp = np.interp(dT_exp, dT_ann, q_ann)

    pct_error = np.abs((q_exp - q_ann_interp) / q_exp) * 100
    mape = np.mean(pct_error)
    return mape


def compute_rmse_htc(dT_ann, htc_ann, dT_exp, htc_exp):
    htc_ann_interp = np.interp(dT_exp, dT_ann, htc_ann)
    rmse = np.sqrt(np.mean((htc_exp - htc_ann_interp) ** 2))
    return rmse


def compute_percent_error_htc(dT_ann, htc_ann, dT_exp, htc_exp):
    htc_ann_interp = np.interp(dT_exp, dT_ann, htc_ann)
    pct_error = np.abs((htc_exp - htc_ann_interp) / htc_exp) * 100
    mape = np.mean(pct_error)
    return mape


def plot(exp_csv, sim_csv, output_path, output_dir, config):
    dT_exp_p = q_exp_p = htc_exp = np.array([])
    dT_sim_p = q_sim_p = htc_sim = np.array([])
    try:
        df_exp = pd.read_csv(exp_csv, header=None, usecols=[0, 1], names=["dT", "q"])
        dT_exp_p, q_exp_p = clean_for_plot(df_exp["dT"].values, df_exp["q"].values)

        htc_exp = htc(q_exp_p, dT_exp_p)
    except FileNotFoundError:
        print("Experiment CSV not found.")

    try:
        df_sim = pd.read_csv(sim_csv, header=None)
        dT_rep = df_sim.iloc[:, 0:10].to_numpy(dtype=float)
        q_sim_raw = df_sim.iloc[:, 10].to_numpy(dtype=float)
        dT_sim_p, q_sim_p = clean_for_plot(np.mean(dT_rep, axis=1), q_sim_raw)

        htc_sim = htc(q_sim_p, dT_sim_p)
    except FileNotFoundError:
        print("Simulation CSV not found.")

    df_ann = pd.read_csv(output_path, header=None, usecols=[0, 1], names=["dT", "q"])
    dT_ann_p, q_ann_p = clean_for_plot(df_ann["dT"].values, df_ann["q"].values)
    dT_li = np.linspace(dT_ann_p.min(), dT_ann_p.max(), 200)
    q_li_Wm2 = liCorreleation(dT_li, config)
    q_li_p = q_li_Wm2 / 1e4  # W/m² -> W/cm²

    if q_exp_p.size > 0:
        q_lo, q_hi = q_exp_p.min(), q_exp_p.max()

        ann_mask = (q_ann_p >= q_lo) & (q_ann_p <= q_hi)
        dT_ann_p = dT_ann_p[ann_mask]
        q_ann_p = q_ann_p[ann_mask]
        htc_ann = htc(q_ann_p, dT_ann_p)

        li_mask = (q_li_p >= q_lo) & (q_li_p <= q_hi)
        dT_li = dT_li[li_mask]
        q_li_p = q_li_p[li_mask]

        if q_sim_p.size > 0:
            sim_mask = (q_sim_p >= q_lo) & (q_sim_p <= q_hi)
            dT_sim_p = dT_sim_p[sim_mask]
            q_sim_p = q_sim_p[sim_mask]
            htc_sim = htc(q_sim_p, dT_sim_p)

    htc_ann = htc(q_ann_p, dT_ann_p)

    htc_li = htc(q_li_p, dT_li)

    if np.size(dT_exp_p) > 1 and np.size(dT_ann_p) > 1:
        rmse = compute_rmse(dT_ann_p, q_ann_p, dT_exp_p, q_exp_p)
        mape = compute_percent_error(dT_ann_p, q_ann_p, dT_exp_p, q_exp_p)
        print(f"RMSE q ({config.name}): {rmse:.2f} W/cm^2")
        print(f"MAPE q ({config.name}): {mape:.2f}%")
        rmse_htc = compute_rmse_htc(dT_ann_p, htc_ann, dT_exp_p, htc_exp)
        mape_htc = compute_percent_error_htc(dT_ann_p, htc_ann, dT_exp_p, htc_exp)
        print(f"RMSE htc ({config.name}): {rmse_htc:.4f} W/(cm^2 K)")
        print(f"MAPE htc ({config.name}): {mape_htc:.2f}%")

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
        label="Experiment (" + config.name + ")",
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

    ax.plot(
        dT_li,
        q_li_p,
        color="#8b3a9e",
        linewidth=2.0,
        zorder=6,
        label="Li et al. (2014)",
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

    ax.text(
        0.98,
        0.02,
        rf"$\phi$:{config.phi_deg}, $k_s$:{config.k_s}, $\rho_s$:{config.rho_s}, $c_{'{'}p,s{'}'}$:{config.c_ps}, ra:{config.ra}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.savefig(
        Path(output_dir) / "qvsdT.jpg",
        dpi=200,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")

    # Experiment
    ax.scatter(
        q_exp_p,
        htc_exp,
        color=COLORS["exp"],
        s=28,
        zorder=4,
        marker="o",
        linewidths=0.4,
        edgecolors="white",
        label="Experiment (" + config.name + ")",
    )

    # Automata simulation
    ax.scatter(
        q_sim_p,
        htc_sim,
        color=COLORS["sim"],
        s=28,
        zorder=3,
        marker="s",
        linewidths=0.4,
        edgecolors="white",
        label="Automata simulation",
    )

    ax.plot(
        q_li_p,
        htc_li,
        color="#8b3a9e",
        linewidth=2.0,
        zorder=6,
        label="Li et al. (2014)",
    )

    # ANN prediction — line + light fill for confidence feel
    ax.plot(
        q_ann_p,
        htc_ann,
        color=COLORS["ann"],
        linewidth=2.0,
        zorder=5,
        label="ANN",
        solid_capstyle="round",
    )
    ax.fill_between(
        q_ann_p,
        htc_ann * 0.85,
        htc_ann * 1.15,
        color=COLORS["ann"],
        alpha=0.10,
        zorder=2,
        label="ANN ±15%",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylabel(r"HTC (W cm$^{-2}$K$^{-1}$) ", labelpad=7)
    ax.set_xlabel(r"Heat flux, $q''$ (W cm$^{-2}$)", labelpad=7)

    ax.tick_params(which="both", top=True, right=True)

    ax.legend()

    ax.text(
        0.98,
        0.02,
        rf"$\phi$:{config.phi_deg}, $k_s$:{config.k_s}, $\rho_s$:{config.rho_s}, $c_{'{'}p,s{'}'}$:{config.c_ps}, ra:{config.ra}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.savefig(
        Path(output_dir) / "htcvsq.jpg",
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
        config.k_s,
        config.rho_s,
        config.c_ps,
        config.ra,
        r_d,
        C1,
        bubble_vol_factor,
    )

    dT = predict(feature_df, model, scaler_x, scaler_y)

    output_path = Path(config.output_dir) / config.output_filename
    save(dT, q_Wcm2_array, output_path)

    plot(config.exp_csv, config.sim_csv, output_path, config.output_dir, config)


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
        generate_boiling_curve(config)
