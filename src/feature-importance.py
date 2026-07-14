"""
Feature Importance Analysis for Neural Networks.

This guide demonstrates 4 methods to compute feature importance:
1. Permutation Importance (model-agnostic, robust)
2. Gradient-based Importance (fast, differentiable)
3. Correlation Analysis (simple baseline)
4. SHAP Values (state-of-the-art, but slower)

For your boiling heat flux model, we'll compute which features matter most.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

from model import BoilingANN

# ============================================================================
# METHOD 1: PERMUTATION FEATURE IMPORTANCE
# ============================================================================


def permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y_true: np.ndarray,
    scaler_X,
    scaler_y,
    feature_names: List[str],
    n_repeats: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance.

    How it works:
    1. Get baseline model performance (e.g., MSE loss)
    2. For each feature:
       - Randomly shuffle that feature
       - Measure performance drop
       - Larger drop = more important
    3. Repeat n_repeats times to get variance estimate

    Pros:
    ✓ Model-agnostic (works with any model)
    ✓ Directly measures impact on target
    ✓ Intuitive interpretation
    ✓ Can use any metric

    Cons:
    ✗ Slower (requires many forward passes)
    ✗ Assumes features are independent

    Args:
        model: Trained neural network
        X: Features (normalized)
        y_true: True target values (log scale)
        scaler_X, scaler_y: Fitted scalers
        feature_names: List of feature names
        n_repeats: Number of repeats for variance
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with importance scores and std
    """
    np.random.seed(random_seed)

    model.eval()

    # Compute baseline loss
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)

    with torch.no_grad():
        baseline_pred = model(X_tensor).numpy()
        baseline_loss = np.mean((baseline_pred - y_true) ** 2)

    print(f"Baseline MSE Loss: {baseline_loss:.6f}")

    # Compute importance for each feature
    importances = []

    for feature_idx, feature_name in enumerate(feature_names):
        losses = []

        for _ in range(n_repeats):
            # Shuffle this feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = np.random.permutation(
                X_permuted[:, feature_idx]
            )
            X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32)

            # Measure loss on permuted data
            with torch.no_grad():
                pred = model(X_perm_tensor).numpy()
                loss = np.mean((pred - y_true) ** 2)

            # Importance = increase in loss
            importance = loss - baseline_loss
            losses.append(importance)

        importances.append(
            {
                "feature": feature_name,
                "importance": np.mean(losses),
                "std": np.std(losses),
            }
        )

    return pd.DataFrame(importances).sort_values("importance", ascending=False)


# ============================================================================
# METHOD 2: GRADIENT-BASED IMPORTANCE
# ============================================================================


def gradient_based_importance(
    model: nn.Module,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    method: str = "mean_abs_grad",
) -> pd.DataFrame:
    """
    Compute gradient-based feature importance.

    How it works:
    1. Compute gradients of output w.r.t. each input feature
    2. Summarize gradients (mean absolute value, mean squared, etc.)
    3. Larger gradient = more sensitive to that feature

    Pros:
    ✓ Very fast (single forward + backward pass)
    ✓ Based on local sensitivity
    ✓ Works for any differentiable model

    Cons:
    ✗ Only measures local gradient, not global impact
    ✗ Gradients can be small near optimal solution
    ✗ Doesn't account for feature interactions

    Args:
        model: Trained neural network
        X: Features (normalized)
        y_true: True target values
        feature_names: List of feature names
        method: "mean_abs_grad", "mean_sq_grad", or "mean_grad"

    Returns:
        DataFrame with importance scores
    """
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)

    # Forward pass
    predictions = model(X_tensor)

    # Compute loss
    loss = torch.mean((predictions - y_tensor) ** 2)

    # Backward pass
    loss.backward()

    # Get gradients
    gradients = X_tensor.grad.numpy()  # Shape: (n_samples, n_features)

    # Summarize gradients
    if method == "mean_abs_grad":
        importance_scores = np.mean(np.abs(gradients), axis=0)
    elif method == "mean_sq_grad":
        importance_scores = np.mean(gradients**2, axis=0)
    elif method == "mean_grad":
        importance_scores = np.mean(np.abs(gradients), axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to sum to 1
    importance_scores = importance_scores / importance_scores.sum()

    importances = []
    for idx, feature_name in enumerate(feature_names):
        importances.append(
            {
                "feature": feature_name,
                "importance": importance_scores[idx],
            }
        )

    return pd.DataFrame(importances).sort_values("importance", ascending=False)


# ============================================================================
# METHOD 3: CORRELATION-BASED IMPORTANCE
# ============================================================================


def correlation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute correlation-based feature importance.

    How it works:
    1. Compute correlation between each feature and target
    2. Absolute correlation = importance

    Pros:
    ✓ Very fast (no model required)
    ✓ Simple and interpretable
    ✓ Good baseline

    Cons:
    ✗ Only measures linear relationships
    ✗ Doesn't account for feature interactions
    ✗ Can't detect non-monotonic relationships

    Args:
        X: Features (normalized)
        y: Target values
        feature_names: List of feature names

    Returns:
        DataFrame with correlation-based importance
    """
    correlations = []

    for idx, feature_name in enumerate(feature_names):
        corr = np.corrcoef(X[:, idx], y)[0, 1]
        correlations.append(
            {
                "feature": feature_name,
                "correlation": abs(corr),
                "raw_corr": corr,
            }
        )

    return pd.DataFrame(correlations).sort_values("correlation", ascending=False)


# ============================================================================
# METHOD 4: SHAP VALUES (ADVANCED)
# ============================================================================


def shap_importance(
    model: nn.Module,
    X: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Compute SHAP-inspired feature importance using gradient-based method.

    Note: This is a simplified approximation. For true SHAP values, use the
    `shap` library: pip install shap

    How it works (simplified):
    1. For each feature, compute contribution by perturbing it
    2. Average across different background samples
    3. Similar to Shapley values from game theory

    Pros:
    ✓ Theoretically sound (based on Shapley values)
    ✓ Provides local + global explanations
    ✓ Accounts for feature interactions

    Cons:
    ✗ Computationally expensive
    ✗ Requires background dataset

    Args:
        model: Trained neural network
        X: Features (background dataset, subset)
        feature_names: List of feature names
        n_samples: Number of background samples

    Returns:
        DataFrame with mean absolute SHAP values
    """
    model.eval()

    # Use subset of data
    X_subset = X[:n_samples]

    shap_values = np.zeros((X_subset.shape[0], X_subset.shape[1]))

    print("Computing SHAP-inspired importance (this may take a moment)...")

    for i, x in enumerate(X_subset):
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Progress: {i+1}/{n_samples}")

        x_tensor = torch.tensor(
            x.reshape(1, -1), dtype=torch.float32, requires_grad=True
        )
        pred = model(x_tensor)

        # Compute gradient for this sample
        pred.backward()
        gradient = x_tensor.grad.numpy().flatten()

        # SHAP value ≈ gradient * input (simplified)
        shap_values[i] = gradient * x

    # Mean absolute SHAP value per feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)

    # Normalize
    mean_shap = mean_shap / mean_shap.sum()

    importances = []
    for idx, feature_name in enumerate(feature_names):
        importances.append(
            {
                "feature": feature_name,
                "shap_importance": mean_shap[idx],
            }
        )

    return pd.DataFrame(importances).sort_values("shap_importance", ascending=False)


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_feature_importance(
    results: Dict[str, pd.DataFrame],
    output_path: str = "results/feature_importance.png",
) -> None:
    """
    Create comparison plot of all importance methods.

    Args:
        results: Dict with method names as keys and DataFrames as values
        output_path: Where to save the plot
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))

    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, df) in zip(axes, results.items()):
        # Get the importance column (changes by method)
        importance_col = [col for col in df.columns if col != "feature"][0]

        # Plot
        ax.barh(df["feature"], df[importance_col], color="steelblue", alpha=0.7)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{method_name}")
        ax.invert_yaxis()

        # Add values on bars
        for i, (feat, val) in enumerate(zip(df["feature"], df[importance_col])):
            ax.text(val, i, f" {val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")


def plot_individual_importance(
    df: pd.DataFrame,
    method_name: str,
    output_path: str = "results/feature_importance_detail.png",
) -> None:
    """Create detailed bar plot for single method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    importance_col = [col for col in df.columns if col != "feature"][0]

    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    ax.bar(df["feature"], df[importance_col], color=colors, alpha=0.8)

    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_title(f"Feature Importance: {method_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Rotate labels
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for i, (feat, val) in enumerate(zip(df["feature"], df[importance_col])):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved: {output_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_feature_importance(
    model_path: str = "models/best_model.pt",
    scaler_x_path: str = "models/scaler_x.pkl",
    scaler_y_path: str = "models/scaler_y.pkl",
    feature_names_path: str = "models/feature_names.json",
    data_dir: str = "data",
    device: str = "cpu",
) -> None:
    """
    Complete feature importance analysis.

    Compares 4 different methods and creates visualizations.
    """

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # ========================================================================
    # LOAD DATA AND MODEL
    # ========================================================================

    print("\nLoading data and model...")

    # Load data
    X_val = np.load(Path(data_dir) / "x_val.npy")
    y_val = np.load(Path(data_dir) / "y_val.npy")

    print(f"  Data shape: {X_val.shape}")
    print(f"  Samples: {len(X_val)}")

    # Load model and scalers
    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    with open(scaler_x_path, "rb") as f:
        scaler_X = pickle.load(f)

    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    model = BoilingANN(len(feature_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    print(f"  Features: {feature_names}")

    # ========================================================================
    # METHOD 1: PERMUTATION IMPORTANCE
    # ========================================================================

    print("\n" + "-" * 70)
    print("METHOD 1: PERMUTATION IMPORTANCE")
    print("-" * 70)
    print("Permuting each feature and measuring impact on loss...")

    perm_importance = permutation_importance(
        model, X_val, y_val, scaler_X, scaler_y, feature_names, n_repeats=10
    )

    print("\nResults:")
    print(perm_importance.to_string(index=False))

    # ========================================================================
    # METHOD 2: GRADIENT-BASED IMPORTANCE
    # ========================================================================

    print("\n" + "-" * 70)
    print("METHOD 2: GRADIENT-BASED IMPORTANCE")
    print("-" * 70)
    print("Computing gradients of output w.r.t. inputs...")

    grad_importance = gradient_based_importance(
        model, X_val, y_val, feature_names, method="mean_abs_grad"
    )

    print("\nResults:")
    print(grad_importance.to_string(index=False))

    # ========================================================================
    # METHOD 3: CORRELATION ANALYSIS
    # ========================================================================

    print("\n" + "-" * 70)
    print("METHOD 3: CORRELATION ANALYSIS")
    print("-" * 70)
    print("Computing correlation between features and target...")

    corr_importance = correlation_importance(X_val, y_val, feature_names)

    print("\nResults:")
    for _, row in corr_importance.iterrows():
        print(
            f"  {row['feature']:20s}: {row['correlation']:6.4f} (raw: {row['raw_corr']:6.4f})"
        )

    # ========================================================================
    # METHOD 4: SHAP-INSPIRED IMPORTANCE
    # ========================================================================

    print("\n" + "-" * 70)
    print("METHOD 4: SHAP-INSPIRED IMPORTANCE (simplified)")
    print("-" * 70)

    shap_imp = shap_importance(model, X_val, feature_names, n_samples=100)

    print("\nResults:")
    print(shap_imp.to_string(index=False))

    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================

    print("\n" + "-" * 70)
    print("SUMMARY AND INTERPRETATION")
    print("-" * 70)

    # Normalize all for comparison
    perm_norm = perm_importance.copy()
    perm_norm["importance"] = perm_norm["importance"] / perm_norm["importance"].sum()

    grad_norm = grad_importance.copy()
    grad_norm.columns = ["feature", "importance"]

    corr_norm = corr_importance[["feature", "correlation"]].copy()
    corr_norm.columns = ["feature", "importance"]

    shap_norm = shap_imp.copy()
    shap_norm.columns = ["feature", "importance"]

    # Ranking agreement
    print("\nRanking by each method:")
    print("\n1. Permutation Importance (most robust):")
    for i, row in perm_importance.head(3).iterrows():
        print(
            f"   {i+1}. {row['feature']:20s}: {row['importance']:.6f} ± {row['std']:.6f}"
        )

    print("\n2. Gradient-Based (fastest):")
    for i, row in grad_importance.head(3).iterrows():
        print(f"   {i+1}. {row['feature']:20s}: {row['importance']:.6f}")

    print("\n3. Correlation (simple baseline):")
    for i, row in corr_importance.head(3).iterrows():
        print(f"   {i+1}. {row['feature']:20s}: {row['correlation']:.6f}")

    print("\n4. SHAP-Inspired:")
    for i, row in shap_imp.head(3).iterrows():
        print(f"   {i+1}. {row['feature']:20s}: {row['shap_importance']:.6f}")

    # Create visualizations
    print("\nCreating visualizations...")

    results_dict = {
        "Permutation": perm_norm[["feature", "importance"]],
        "Gradient": grad_norm,
        "Correlation": corr_norm,
        "SHAP": shap_norm,
    }

    plot_feature_importance(results_dict)
    plot_individual_importance(perm_importance, "Permutation Importance")

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)

    # Save results
    with open("results/feature_importance_summary.txt", "w") as f:
        f.write("FEATURE IMPORTANCE ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("METHOD 1: PERMUTATION IMPORTANCE\n")
        f.write(perm_importance.to_string(index=False))
        f.write("\n\nMETHOD 2: GRADIENT-BASED\n")
        f.write(grad_importance.to_string(index=False))
        f.write("\n\nMETHOD 3: CORRELATION\n")
        f.write(corr_importance.to_string(index=False))
        f.write("\n\nMETHOD 4: SHAP-INSPIRED\n")
        f.write(shap_imp.to_string(index=False))

    print("\n✓ Results saved to results/feature_importance_summary.txt")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    analyze_feature_importance()
