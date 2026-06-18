import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatter, NullFormatter
import matplotlib.ticker as ticker
import numpy as np

exp_csv = "data/experiment/jiang2021_ex.csv"
sim_csv = "data/automata/jiang2021_sim_benjamin.csv"
ann_csv = "data/simulation/jiang2021_ann.csv"


def clean_for_plot(dT, q):
    dT = np.asarray(dT, dtype=float)
    q = np.asarray(q, dtype=float)
    m = (dT > 0) & (q > 0) & np.isfinite(dT) & np.isfinite(q)
    dT2, q2 = dT[m], q[m]
    idx = np.argsort(dT2)
    return dT2[idx], q2[idx]


# ── Load data ─────────────────────────────────────────────────────────────────
df_exp = pd.read_csv(exp_csv, header=None, usecols=[0, 1], names=["dT", "q"])
dT_exp_p, q_exp_p = clean_for_plot(df_exp["dT"].values, df_exp["q"].values)

df_sim = pd.read_csv(sim_csv, header=None)
dT_rep = df_sim.iloc[:, 0:10].to_numpy(dtype=float)
q_sim_raw = df_sim.iloc[:, 10].to_numpy(dtype=float)
dT_sim_p, q_sim_p = clean_for_plot(np.mean(dT_rep, axis=1), q_sim_raw)

df_ann = pd.read_csv(ann_csv, header=None, usecols=[0, 1], names=["dT", "q"])
dT_ann_p, q_ann_p = clean_for_plot(df_ann["dT"].values, df_ann["q"].values)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        # "font.size": 11,
        # "axes.linewidth": 1.2,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # "xtick.direction": "in",
        # "ytick.direction": "in",
        # "xtick.major.size": 5,
        # "ytick.major.size": 5,
        # "xtick.minor.size": 3,
        # "ytick.minor.size": 3,
        # "xtick.major.width": 1.1,
        # "ytick.major.width": 1.1,
        # "legend.frameon": True,
        # "legend.framealpha": 0.92,
        # "legend.edgecolor": "#cccccc",
        # "legend.fontsize": 10,
        # "figure.dpi": 200,
    }
)

COLORS = {
    "exp": "#1a1a2e",  # deep navy
    "sim": "#e07b39",  # warm amber
    "ann": "#2e7d59",  # forest green
}

# ── Figure ────────────────────────────────────────────────────────────────────
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

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"Wall superheat, $\Delta T$ (K)", labelpad=7)
ax.set_ylabel(r"Heat flux, $q''$ (W cm$^{-2}$)", labelpad=7)

# Clean log tick labels (show only select major ticks)
# ax.xaxis.set_major_formatter(
#   ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
# )
# ax.yaxis.set_major_formatter(
#    ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
# )
# ax.xaxis.set_minor_formatter(NullFormatter())
# ax.yaxis.set_minor_formatter(NullFormatter())

ax.tick_params(which="both", top=True, right=True)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(
    loc="upper left",
)

plt.savefig("loglog.jpg", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig("loglog.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: loglog.jpg  loglog.pdf")
