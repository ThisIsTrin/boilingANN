import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CONFIG_DIR = Path("configs/sweep")
MANIFEST_PATH = Path("configs/sweep_manifest.json")

CELL_THICKNESS_M = 0.05
TIMESTEP_S = 0.005
TIMESTEP_SAFETY_FACTOR = (
    1.5  # margin applied on top of the measured 99%-convergence timestep
)
LEGACY_FIXED_TIMESTEPS = (
    40001  # previous uniform value, kept only for the before/after comparison
)

CONTACT_ANGLES_DEG = [
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
    110,
]

ROUGHNESS_RA_M = [0.01e-6, 0.05e-6, 0.1e-6, 0.5e-6, 1e-6, 5e-6]
SEEDS = [0]


@dataclass(frozen=True)
class Material:
    name: str
    thermal_conductivity: float  # W/(m*K)
    thermal_diffusivity: float  # m^2/s
    emissivity: float
    heat_flux_wcm2: list[int]  # heat fluxes to sweep, in W/cm^2
    convergence_99pct_timesteps: dict[
        int, int
    ]  # q [W/cm^2] -> measured 99%-convergence timestep


MATERIALS = [
    Material(
        name="SiO2",
        thermal_conductivity=1.4,
        thermal_diffusivity=6.38e-7,
        emissivity=0.8,
        heat_flux_wcm2=[1, 5, 10, 25, 50, 75, 100],
        convergence_99pct_timesteps={
            1: 21237,
            5: 6938,
            10: 9702,
            25: 8521,
            50: 7700,
            75: 7102,
            100: 6624,
        },
    ),
    Material(
        name="Copper",
        thermal_conductivity=393.0,
        thermal_diffusivity=1.1082e-4,
        emissivity=0.03,
        heat_flux_wcm2=[1, 5, 10, 25, 50, 75, 100, 150],
        convergence_99pct_timesteps={
            1: 28448,
            5: 11244,
            10: 15740,
            25: 13868,
            50: 12681,
            75: 11924,
            100: 11351,
            150: 10620,
        },
    ),
]


def adaptive_max_timesteps(material: Material, q_wcm2: int) -> int:
    # Round the safety-padded convergence timestep up to the nearest 1000.
    raw = material.convergence_99pct_timesteps[q_wcm2] * TIMESTEP_SAFETY_FACTOR
    return math.ceil(raw / 1000) * 1000


def wcm2_to_wm2(q_wcm2: int) -> int:
    return int(q_wcm2 * 1e4)


def build_config(
    *,
    material: Material,
    phi_deg: int,
    q_wcm2: int,
    roughness_ra: float,
    seed: int,
) -> tuple[str, str, dict]:
    # Build a single run's config dict. Returns (sim_name, output_path, config).
    q_wm2 = wcm2_to_wm2(q_wcm2)  # heat_gen_initial is a surface flux, in W/m^2
    max_timesteps = adaptive_max_timesteps(material, q_wcm2)

    sim_name = f"{material.name}_phi{phi_deg}_q{q_wm2}_r{roughness_ra}_seed{seed}"
    output_path = f"output/sweep/{sim_name}"

    config = {
        "metadata": {
            "sim_name": sim_name,
            "output_path": output_path,
            "write_freq": 1,
            "seed": seed,
        },
        "simulation": {
            "dt": TIMESTEP_S,
            "max_bubbles": 10000,
            "maximum_timesteps": max_timesteps,
        },
        "heater_automata": {
            "cell_counts": (10, 10),
            "cell_length": 0.002215,
            "cell_thickness": CELL_THICKNESS_M,
            "num_nucleation_points": 100,
            "heat_gen_initial": q_wm2,  # [W/m^2]
            "heat_gen_increase": 0,
            "heat_gen_interval": 123,
            "heater_thermal_conductivity": material.thermal_conductivity,
            "heater_thermal_diffusivity": material.thermal_diffusivity,
            "heater_emissivity": material.emissivity,
            "contact_angle_mean": phi_deg,
            "contact_angle_stdv": 0,
            "bulk_temperature": 373.15,
            "saturation_temperature": 373.15,
            "bulk_pressure": 0.101325,
            "natural_convection_constant": 1.0,
            "microconvection_coefficient": 1.0,
            "surface_coalescence": False,
            "nucleation_model": {
                "name": "benjamin",
                "surface_roughness_Ra": roughness_ra,
            },
        },
        "bubble_automata": {
            "enabled": True,
            "domain_bounds": (-0.1, 0.1, -0.1, 0.1),
            "critical_radius": 0.0018,
            "drag_model_name": None,
            "bubble_grow_model": "cooper_lloyd_modified",
            "bubble_detach_model": "fritz",
            "contamination_level": 2,
            "buoyancy_displacement": 0.005,
            "turbulence_displacement": 0.001,
        },
    }

    return sim_name, output_path, config


def iter_runs():
    # Yield every (material, phi, q_wcm2, roughness, seed) combination to sweep.
    for material in MATERIALS:
        for phi_deg, q_wcm2, roughness_ra, seed in itertools.product(
            CONTACT_ANGLES_DEG, material.heat_flux_wcm2, ROUGHNESS_RA_M, SEEDS
        ):
            yield material, phi_deg, q_wcm2, roughness_ra, seed


def log_adaptive_timestep_table() -> None:
    log.info("Adaptive maximum_timesteps per q level:")
    for material in MATERIALS:
        for q_wcm2 in material.heat_flux_wcm2:
            log.info(
                "  %-6s q=%4d W/cm^2 -> %6d ts (was %d)",
                material.name,
                q_wcm2,
                adaptive_max_timesteps(material, q_wcm2),
                LEGACY_FIXED_TIMESTEPS,
            )


def generate_all() -> list[dict]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for run_index, (material, phi_deg, q_wcm2, roughness_ra, seed) in enumerate(
        iter_runs()
    ):
        sim_name, output_path, config = build_config(
            material=material,
            phi_deg=phi_deg,
            q_wcm2=q_wcm2,
            roughness_ra=roughness_ra,
            seed=seed,
        )

        config_path = CONFIG_DIR / f"{sim_name}.py"
        config_path.write_text(repr(config))

        manifest.append(
            {
                "run_index": run_index,
                "sim_name": sim_name,
                "config_path": str(config_path),
                "output_path": output_path,
                "material": material.name,
                "phi": phi_deg,
                "q_wm2": config["heater_automata"]["heat_gen_initial"],
                "q_Wcm2": q_wcm2,
                "max_timesteps": config["simulation"]["maximum_timesteps"],
                "k_s": material.thermal_conductivity,
                "alpha_s": material.thermal_diffusivity,
                "ra": roughness_ra,
                "seed": seed,
            }
        )

    return manifest


def main() -> None:
    log_adaptive_timestep_table()

    manifest = generate_all()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=4))

    total_ts = sum(r["max_timesteps"] for r in manifest)
    old_total_ts = len(manifest) * LEGACY_FIXED_TIMESTEPS

    log.info("")
    log.info("Generated %d configs in %s/", len(manifest), CONFIG_DIR)
    log.info("Total timesteps  : %s", f"{total_ts:,}")
    log.info("Old uniform total: %s", f"{old_total_ts:,}")
    log.info("Change: %+.0f%%", (total_ts / old_total_ts - 1) * 100)


if __name__ == "__main__":
    main()
