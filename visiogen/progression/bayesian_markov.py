"""
bayesian_markov.py
==================
Bayesian Markov Chain model for eye disease progression.

Models disease as a sequence of stages with transition probabilities
that are treated as Beta-distributed random variables (not fixed scalars).
Genetic risk score and lifestyle covariates modulate transition rates.

Anchor disease: Age-related Macular Degeneration (AMD)
  Stages: Healthy → Early AMD → Intermediate AMD → Advanced AMD → Legal Blindness

Usage:
    from bayesian_markov import BayesianMarkovModel
    model = BayesianMarkovModel(disease="AMD")
    trajectory = model.simulate_trajectory(risk_score=0.72, age_start=40, age_end=80)
    model.plot_trajectory(trajectory)

Author: VisioGen Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import beta as beta_dist
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Disease configuration
# ---------------------------------------------------------------------------

DISEASE_CONFIGS = {
    "AMD": {
        "stages": [
            "Healthy",
            "Early AMD",
            "Intermediate AMD",
            "Advanced AMD",
            "Legal Blindness",
        ],
        "stage_colors": ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
        # Base transition probabilities (per year) between consecutive stages.
        # Source priors drawn from:
        #   - Ferris et al. (2013) Age-Related Eye Disease Study
        #   - Joachim et al. (2017) meta-analysis
        # Each tuple is (alpha, beta) of the Beta prior — mean = alpha/(alpha+beta)
        "base_transitions": [
            (1.5, 18.5),   # Healthy       → Early AMD       ~7.5% / yr
            (2.0, 18.0),   # Early AMD      → Intermediate    ~10% / yr
            (3.0, 17.0),   # Intermediate   → Advanced        ~15% / yr
            (2.5, 17.5),   # Advanced       → Legal Blindness ~12.5% / yr
        ],
        # How much the risk score scales transition rates (multiplicative)
        # A risk_score of 0 → multiplier of 1 (baseline)
        # A risk_score of 1 → multiplier of (1 + max_amplification)
        "max_amplification": 2.5,
    },
    "Glaucoma": {
        "stages": [
            "Healthy",
            "Suspected Glaucoma",
            "Early Glaucoma",
            "Moderate Glaucoma",
            "Advanced Glaucoma",
            "Blindness",
        ],
        "stage_colors": ["#2ecc71", "#a8e6cf", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
        "base_transitions": [
            (0.5, 19.5),   # Healthy       → Suspected       ~2.5% / yr
            (1.5, 18.5),   # Suspected     → Early           ~7.5% / yr
            (2.0, 18.0),   # Early         → Moderate        ~10% / yr
            (1.5, 18.5),   # Moderate      → Advanced        ~7.5% / yr
            (1.0, 19.0),   # Advanced      → Blindness       ~5% / yr
        ],
        "max_amplification": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TransitionPrior:
    """
    Beta distribution prior on a single stage transition probability.

    The Beta(alpha, beta) distribution is the natural conjugate prior
    for a Bernoulli probability. Mean = alpha / (alpha + beta).
    """
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta
        return np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

    def sample(self, n: int = 1) -> np.ndarray:
        return beta_dist.rvs(self.alpha, self.beta, size=n)

    def __repr__(self):
        return f"Beta(α={self.alpha}, β={self.beta}) | mean={self.mean:.3f}, std={self.std:.3f}"


@dataclass
class Trajectory:
    """
    Output of a simulated disease trajectory.

    Attributes:
        ages           : array of ages simulated (e.g. 30..80)
        stage_probs    : (n_ages, n_stages) array — probability of being in each stage at each age
        stage_probs_ci : dict with 'lower' and 'upper' (n_ages, n_stages) confidence interval arrays
        risk_score     : genetic risk score used
        disease        : disease name
        n_simulations  : number of Monte Carlo runs used
    """
    ages: np.ndarray
    stage_probs: np.ndarray
    stage_probs_ci: dict
    risk_score: float
    disease: str
    n_simulations: int
    stage_names: list[str]
    stage_colors: list[str]


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class BayesianMarkovModel:
    """
    Bayesian Markov Chain model for progressive eye disease.

    Transition probabilities are Beta-distributed random variables.
    Uncertainty is propagated via Monte Carlo simulation.

    Parameters
    ----------
    disease : str
        One of the keys in DISEASE_CONFIGS (e.g. "AMD", "Glaucoma").
    """

    def __init__(self, disease: str = "AMD"):
        if disease not in DISEASE_CONFIGS:
            raise ValueError(f"Disease '{disease}' not found. Available: {list(DISEASE_CONFIGS.keys())}")

        self.disease = disease
        config = DISEASE_CONFIGS[disease]
        self.stages = config["stages"]
        self.stage_colors = config["stage_colors"]
        self.n_stages = len(self.stages)
        self.max_amplification = config["max_amplification"]

        # Build transition priors
        self.priors: list[TransitionPrior] = [
            TransitionPrior(a, b) for a, b in config["base_transitions"]
        ]

    # ------------------------------------------------------------------
    # Risk modulation
    # ------------------------------------------------------------------

    def _amplify_transition(self, base_prob: float, risk_score: float) -> float:
        """
        Scale a transition probability by the genetic risk score.

        Uses a multiplicative model:
            p_adjusted = p_base * (1 + amplification * risk_score)

        Clipped to [0, 1].

        Parameters
        ----------
        base_prob  : float in [0, 1]  — baseline transition probability
        risk_score : float in [0, 1]  — polygenic risk score
        """
        multiplier = 1.0 + self.max_amplification * risk_score
        return float(np.clip(base_prob * multiplier, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Single simulation run
    # ------------------------------------------------------------------

    def _single_run(
        self,
        risk_score: float,
        age_start: int,
        age_end: int,
        sampled_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Run one Markov chain simulation using a fixed set of sampled transition probs.

        Returns
        -------
        stage_indices : np.ndarray of shape (n_years,)
            Stage index at each year.
        """
        n_years = age_end - age_start + 1
        stage_indices = np.zeros(n_years, dtype=int)
        current_stage = 0  # start healthy

        for t in range(n_years):
            stage_indices[t] = current_stage
            # At terminal stage, stay there
            if current_stage >= self.n_stages - 1:
                continue
            # Transition probability for this stage, adjusted by risk score
            p_transition = self._amplify_transition(
                sampled_probs[current_stage], risk_score
            )
            if np.random.random() < p_transition:
                current_stage += 1

        return stage_indices

    # ------------------------------------------------------------------
    # Monte Carlo trajectory simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        risk_score: float,
        age_start: int = 30,
        age_end: int = 80,
        n_simulations: int = 5000,
        ci: float = 0.95,
        seed: Optional[int] = 42,
    ) -> Trajectory:
        """
        Simulate a disease trajectory with full Bayesian uncertainty.

        For each Monte Carlo run:
          1. Sample transition probabilities from their Beta priors
          2. Run the Markov chain from age_start to age_end
          3. Record the stage at each age

        Aggregate across runs to get posterior probabilities and CIs.

        Parameters
        ----------
        risk_score    : float in [0, 1] — polygenic risk score
        age_start     : int — starting age for simulation
        age_end       : int — ending age
        n_simulations : int — number of Monte Carlo samples
        ci            : float — confidence interval width (e.g. 0.95)
        seed          : int or None — random seed for reproducibility

        Returns
        -------
        Trajectory object with stage probabilities and uncertainty bounds.
        """
        if not 0.0 <= risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")
        if age_start >= age_end:
            raise ValueError("age_start must be less than age_end")

        if seed is not None:
            np.random.seed(seed)

        n_years = age_end - age_start + 1
        ages = np.arange(age_start, age_end + 1)

        # Store one-hot stage occupancy across all runs: (n_sims, n_years, n_stages)
        occupancy = np.zeros((n_simulations, n_years, self.n_stages), dtype=np.float32)

        for sim in range(n_simulations):
            # Sample transition probabilities from priors
            sampled_probs = np.array([prior.sample(1)[0] for prior in self.priors])

            # Run chain
            stage_indices = self._single_run(risk_score, age_start, age_end, sampled_probs)

            # One-hot encode
            for t, s in enumerate(stage_indices):
                occupancy[sim, t, s] = 1.0

        # Posterior mean stage probabilities
        stage_probs = occupancy.mean(axis=0)  # (n_years, n_stages)

        # Confidence intervals across simulations (per stage, per year)
        alpha = (1.0 - ci) / 2.0
        lower = np.quantile(occupancy, alpha, axis=0)
        upper = np.quantile(occupancy, 1.0 - alpha, axis=0)

        return Trajectory(
            ages=ages,
            stage_probs=stage_probs,
            stage_probs_ci={"lower": lower, "upper": upper},
            risk_score=risk_score,
            disease=self.disease,
            n_simulations=n_simulations,
            stage_names=self.stages,
            stage_colors=self.stage_colors,
        )

    # ------------------------------------------------------------------
    # Expected time to each stage
    # ------------------------------------------------------------------

    def expected_time_to_stage(
        self,
        risk_score: float,
        target_stage: int,
        age_start: int = 40,
        age_end: int = 90,
        n_simulations: int = 5000,
        seed: Optional[int] = 42,
    ) -> dict:
        """
        Compute the expected age (and uncertainty) at which a patient
        first reaches a target disease stage.

        Returns
        -------
        dict with keys: 'mean', 'std', 'median', 'never_reached_pct'
        """
        if seed is not None:
            np.random.seed(seed)

        n_years = age_end - age_start + 1
        ages = np.arange(age_start, age_end + 1)
        first_arrival_ages = []

        for _ in range(n_simulations):
            sampled_probs = np.array([prior.sample(1)[0] for prior in self.priors])
            stage_indices = self._single_run(risk_score, age_start, age_end, sampled_probs)

            arrivals = np.where(stage_indices >= target_stage)[0]
            if len(arrivals) > 0:
                first_arrival_ages.append(ages[arrivals[0]])

        never_pct = 100.0 * (1 - len(first_arrival_ages) / n_simulations)

        if len(first_arrival_ages) == 0:
            return {"mean": None, "std": None, "median": None, "never_reached_pct": 100.0}

        arr = np.array(first_arrival_ages)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "never_reached_pct": never_pct,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_trajectory(
        self,
        trajectory: Trajectory,
        figsize: tuple = (12, 6),
        show_ci: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the posterior disease stage probabilities over age,
        with confidence interval shading.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        ages = trajectory.ages
        probs = trajectory.stage_probs
        ci_lower = trajectory.stage_probs_ci["lower"]
        ci_upper = trajectory.stage_probs_ci["upper"]

        for i, (stage, color) in enumerate(
            zip(trajectory.stage_names, trajectory.stage_colors)
        ):
            ax.plot(ages, probs[:, i], color=color, linewidth=2.5, label=stage, zorder=3)
            if show_ci:
                ax.fill_between(
                    ages,
                    ci_lower[:, i],
                    ci_upper[:, i],
                    color=color,
                    alpha=0.12,
                    zorder=2,
                )

        ax.set_xlabel("Age (years)", color="white", fontsize=12)
        ax.set_ylabel("Probability", color="white", fontsize=12)
        ax.set_title(
            f"{trajectory.disease} Progression — Risk Score: {trajectory.risk_score:.2f}",
            color="white",
            fontsize=14,
            fontweight="bold",
        )
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(ages[0], ages[-1])
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper right",
            framealpha=0.2,
            labelcolor="white",
            facecolor="#1a1a1a",
            edgecolor="#444",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
            print(f"Saved to {save_path}")
        plt.show()

    def plot_risk_comparison(
        self,
        risk_scores: list[float],
        target_stage: int,
        age_start: int = 40,
        age_end: int = 80,
        n_simulations: int = 3000,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare progression probability for a target stage across
        multiple genetic risk scores. Useful for visualizing how much
        genetics matters.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        cmap = cm.get_cmap("RdYlGn_r", len(risk_scores))

        for idx, rs in enumerate(risk_scores):
            traj = self.simulate_trajectory(
                risk_score=rs,
                age_start=age_start,
                age_end=age_end,
                n_simulations=n_simulations,
                seed=None,
            )
            color = cmap(idx / max(len(risk_scores) - 1, 1))
            ax.plot(
                traj.ages,
                traj.stage_probs[:, target_stage],
                color=color,
                linewidth=2,
                label=f"Risk = {rs:.2f}",
            )

        stage_name = self.stages[target_stage]
        ax.set_xlabel("Age (years)", color="white", fontsize=12)
        ax.set_ylabel(f"P(≥ {stage_name})", color="white", fontsize=12)
        ax.set_title(
            f"{self.disease}: Probability of Reaching '{stage_name}' by Age\n"
            f"Across Genetic Risk Scores",
            color="white",
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(age_start, age_end)
        ax.set_ylim(0, 1)
        ax.legend(
            loc="upper left",
            framealpha=0.2,
            labelcolor="white",
            facecolor="#1a1a1a",
            edgecolor="#444",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def print_summary(self, trajectory: Trajectory) -> None:
        """Print a readable summary of a simulated trajectory."""
        print(f"\n{'='*55}")
        print(f"  {trajectory.disease} Progression Summary")
        print(f"  Risk Score : {trajectory.risk_score:.3f}")
        print(f"  Age Range  : {trajectory.ages[0]} – {trajectory.ages[-1]}")
        print(f"  Simulations: {trajectory.n_simulations:,}")
        print(f"{'='*55}")
        print(f"  {'Stage':<25} {'P(at age 60)':>12} {'P(at age 75)':>12}")
        print(f"  {'-'*50}")

        idx_60 = int(60 - trajectory.ages[0]) if 60 >= trajectory.ages[0] else 0
        idx_75 = int(75 - trajectory.ages[0]) if 75 >= trajectory.ages[0] else -1

        for i, stage in enumerate(trajectory.stage_names):
            p60 = trajectory.stage_probs[idx_60, i]
            p75 = trajectory.stage_probs[idx_75, i]
            print(f"  {stage:<25} {p60:>11.1%} {p75:>11.1%}")
        print(f"{'='*55}\n")

        # Expected time to advanced stage
        advanced_idx = len(self.stages) - 2
        result = self.expected_time_to_stage(
            risk_score=trajectory.risk_score,
            target_stage=advanced_idx,
            age_start=trajectory.ages[0],
            age_end=trajectory.ages[-1],
            n_simulations=trajectory.n_simulations,
        )
        stage_name = self.stages[advanced_idx]
        if result["mean"] is not None:
            print(f"  Expected age at '{stage_name}':")
            print(f"    Mean   : {result['mean']:.1f} yrs")
            print(f"    Median : {result['median']:.1f} yrs")
            print(f"    Std    : {result['std']:.1f} yrs")
        print(f"    Never reached (in window): {result['never_reached_pct']:.1f}%")
        print()