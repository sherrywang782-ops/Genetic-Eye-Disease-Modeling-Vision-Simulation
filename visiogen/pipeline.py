"""
pipeline.py
===========
VisioGen — End-to-end pipeline connecting genetic risk, disease progression,
and visual simulation into a single personalized output.

Given a (simulated or real) genetic profile and lifestyle inputs, this script:
  1. Computes a Polygenic Risk Score (PRS) with GxE adjustment
  2. Feeds the risk score into a Bayesian Markov progression model
  3. Derives severity scores from the progression trajectory
  4. Renders a visual simulation of how the person's vision changes over time

Usage:
    python pipeline.py

Author: VisioGen Project
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root to path so imports work on all platforms
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from genetics.prs_pipeline import PRSModel, EnvironmentalProfile
from progression.bayesian_markov import BayesianMarkovModel
from simulation.disease_filters import EyeDiseaseSimulator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DISEASE       = "AMD"
AGE_START     = 40
AGE_END       = 80
N_SIMULATIONS = 3000
SEED          = 42


# ---------------------------------------------------------------------------
# Helper: derive severity score from stage probability distribution
# ---------------------------------------------------------------------------

def stage_probs_to_severity(stage_probs: np.ndarray) -> np.ndarray:
    """
    Convert (n_ages, n_stages) stage probability matrix to a
    continuous severity score in [0, 1] per age.

    Uses expected stage index: severity = E[stage] / (n_stages - 1)
    """
    n_stages = stage_probs.shape[1]
    stage_indices = np.arange(n_stages)
    expected_stage = stage_probs @ stage_indices
    return expected_stage / (n_stages - 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    risk_score: float = None,
    env: EnvironmentalProfile = None,
    image=None,
    verbose: bool = True,
):
    """
    Run the full VisioGen pipeline for one individual.

    Parameters
    ----------
    risk_score : float in [0, 1] or None (computed from genotype if None)
    env        : EnvironmentalProfile or None (uses defaults if None)
    image      : np.ndarray or path, or None (uses synthetic sample)
    verbose    : bool — print summaries

    Returns
    -------
    dict with keys: risk_score, trajectory, severity_scores
    """

    print("\n" + "=" * 60)
    print("  VisioGen — Personalized Eye Disease Modeling Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Genetic Risk Score
    # ------------------------------------------------------------------
    print(f"\n[1/3] Computing Polygenic Risk Score ({DISEASE})...")

    prs_model = PRSModel(disease=DISEASE)

    if env is None:
        env = EnvironmentalProfile(age=45, smoking=False, bmi=24.0, uv_exposure=0.4)

    if risk_score is None:
        genotype = prs_model.simulate_genotype(n_individuals=1, seed=SEED)[0]
        risk_score = prs_model.compute_risk(genotype, env)
        if verbose:
            prs_model.print_summary(genotype, env)
    else:
        genotype = prs_model.simulate_genotype(n_individuals=1, seed=SEED)[0]
        if verbose:
            print(f"  Using provided risk score: {risk_score:.3f}")

    print(f"  ✓ Composite risk score: {risk_score:.3f}")

    # ------------------------------------------------------------------
    # Step 2: Bayesian Markov Progression
    # ------------------------------------------------------------------
    print(f"\n[2/3] Simulating disease progression (ages {AGE_START}–{AGE_END})...")

    markov = BayesianMarkovModel(disease=DISEASE)
    trajectory = markov.simulate_trajectory(
        risk_score=risk_score,
        age_start=AGE_START,
        age_end=AGE_END,
        n_simulations=N_SIMULATIONS,
        seed=SEED,
    )

    if verbose:
        markov.print_summary(trajectory)

    severity_scores = stage_probs_to_severity(trajectory.stage_probs)
    print(f"  ✓ Severity at age 60: {severity_scores[60 - AGE_START]:.3f}")
    print(f"  ✓ Severity at age 75: {severity_scores[75 - AGE_START]:.3f}")

    # ------------------------------------------------------------------
    # Step 3: Visual Simulation
    # ------------------------------------------------------------------
    print(f"\n[3/3] Rendering visual simulation...")

    sim = EyeDiseaseSimulator()
    if image is None:
        image = sim.load_sample_image()

    print(f"  ✓ Generating progression frames...")

    # Plot all three outputs
    print("\n  → Plotting progression trajectory...")
    markov.plot_trajectory(trajectory)

    print("  → Rendering vision timeline...")
    sim.show_trajectory_frames(
        image=image,
        disease=DISEASE,
        ages=trajectory.ages,
        severity_scores=severity_scores,
        selected_ages=[40, 50, 60, 70, 80],
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60 + "\n")

    return {
        "risk_score": risk_score,
        "trajectory": trajectory,
        "severity_scores": severity_scores,
    }


def compare_risk_profiles(image=None):
    """
    Compare two individuals — low vs. high genetic risk — side by side.
    Useful for communicating the impact of genetics.
    """
    sim = EyeDiseaseSimulator()
    markov = BayesianMarkovModel(disease=DISEASE)

    if image is None:
        image = sim.load_sample_image()

    low_risk  = 0.15
    high_risk = 0.80

    print(f"\nComparing risk profiles: {low_risk} vs {high_risk}")

    traj_low  = markov.simulate_trajectory(low_risk,  AGE_START, AGE_END, N_SIMULATIONS, seed=SEED)
    traj_high = markov.simulate_trajectory(high_risk, AGE_START, AGE_END, N_SIMULATIONS, seed=SEED)

    sev_low  = stage_probs_to_severity(traj_low.stage_probs)
    sev_high = stage_probs_to_severity(traj_high.stage_probs)

    # Plot progression curves together
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    ax.plot(traj_low.ages,  sev_low,  color="#2ecc71", linewidth=2.5, label=f"Low risk  (PRS={low_risk})")
    ax.plot(traj_high.ages, sev_high, color="#e74c3c", linewidth=2.5, label=f"High risk (PRS={high_risk})")
    ax.fill_between(traj_low.ages, sev_low, sev_high, alpha=0.1, color="#e74c3c")
    ax.set_xlabel("Age (years)", color="white", fontsize=12)
    ax.set_ylabel("Expected Severity", color="white", fontsize=12)
    ax.set_title(f"{DISEASE} — Low vs. High Genetic Risk Trajectory", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444"); ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.legend(labelcolor="white", facecolor="#1a1a1a", edgecolor="#444")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # Vision at age 70: low vs high risk
    age_70_idx = 70 - AGE_START
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="#0f0f0f")
    fig.suptitle(f"Vision at Age 70 — {DISEASE}", color="white", fontsize=13, fontweight="bold")

    for ax, img, title in zip(
        axes,
        [
            image,
            sim.simulate(image, DISEASE, float(sev_low[age_70_idx])),
            sim.simulate(image, DISEASE, float(sev_high[age_70_idx])),
        ],
        [
            "Normal Vision",
            f"Low Risk (sev={sev_low[age_70_idx]:.2f})",
            f"High Risk (sev={sev_high[age_70_idx]:.2f})",
        ],
    ):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with default simulated individual
    result = run_pipeline(
        env=EnvironmentalProfile(age=45, smoking=True, bmi=26.5, uv_exposure=0.55, diabetes=False),
    )

    # Optionally: compare two risk profiles
    compare_risk_profiles()