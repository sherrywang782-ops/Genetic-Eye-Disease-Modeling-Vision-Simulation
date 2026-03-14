"""
prs_pipeline.py
===============
Polygenic Risk Score (PRS) modeling with Gene-Environment (GxE) interaction.

A PRS aggregates the effects of many genetic variants (SNPs) identified
through Genome-Wide Association Studies (GWAS) into a single continuous
risk score per individual.

This module:
  1. Constructs a PRS from GWAS effect sizes and individual genotypes
  2. Standardizes and normalizes the score to [0, 1]
  3. Models gene-environment interaction (GxE): lifestyle/environmental
     factors that modulate the genetic risk
  4. Outputs a final composite risk score for downstream progression modeling

Usage:
    from prs_pipeline import PRSModel, EnvironmentalProfile
    model = PRSModel(disease="AMD")
    env = EnvironmentalProfile(age=55, smoking=True, bmi=27.5, uv_exposure=0.6)
    risk = model.compute_risk(genotype_vector, env)

Data note:
    Real genotype data can be sourced from:
      - UK Biobank (application required)
      - GWAS Catalog for effect sizes: https://www.ebi.ac.uk/gwas/
    For simulation and testing, synthetic genotypes are generated here.

Author: VisioGen Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from scipy.special import expit  # sigmoid function


# ---------------------------------------------------------------------------
# Known AMD risk variants (simplified from literature)
# Sources:
#   Fritsche et al. (2016) Nature Genetics — 52 AMD risk loci
#   Seddon et al. (2016)  — CFH, ARMS2 as strongest signals
# ---------------------------------------------------------------------------

AMD_GWAS_VARIANTS = pd.DataFrame({
    "rsid": [
        "rs1061170",   # CFH Y402H — strongest single AMD locus
        "rs10490924",  # ARMS2 A69S — second strongest
        "rs2230199",   # C3
        "rs9332739",   # CFB
        "rs10737680",  # CFH region
        "rs3750847",   # ARMS2/HTRA1
        "rs1329428",   # CFH
        "rs4698775",   # CFI
        "rs13081855",  # COL8A1
        "rs8135665",   # SLC16A8
        "rs3812111",   # CFH
        "rs429608",    # C2/CFB
        "rs11200638",  # HTRA1
        "rs2736911",   # CETP
        "rs1864163",   # APOE region
    ],
    # Effect size (log-odds ratio from GWAS meta-analysis)
    "effect_size": [
        0.65,   # CFH — large effect
        0.59,   # ARMS2 — large effect
        0.21,
        0.19,
        0.17,
        0.55,   # HTRA1 region
        0.18,
        0.15,
        0.12,
        0.10,
        0.14,
        0.22,
        0.48,   # HTRA1
        0.11,
        0.09,
    ],
    # Risk allele frequency in European-ancestry populations
    "risk_allele_freq": [
        0.35, 0.22, 0.23, 0.12, 0.55,
        0.25, 0.60, 0.38, 0.14, 0.45,
        0.50, 0.18, 0.20, 0.40, 0.15,
    ],
})

GLAUCOMA_GWAS_VARIANTS = pd.DataFrame({
    "rsid": [
        "rs4236601",   # CAV1/CAV2
        "rs7137828",   # TMCO1
        "rs2472493",   # CDKN2B-AS1
        "rs9913911",   # SIX1/SIX6
        "rs1900004",   # TGFBR3
        "rs3213787",   # WDR36
        "rs7588567",   # AFAP1
        "rs2073491",   # GAS7
    ],
    "effect_size": [
        0.18, 0.16, 0.20, 0.22, 0.14, 0.12, 0.15, 0.11
    ],
    "risk_allele_freq": [
        0.45, 0.30, 0.55, 0.40, 0.25, 0.35, 0.20, 0.50
    ],
})

DISEASE_VARIANTS = {
    "AMD": AMD_GWAS_VARIANTS,
    "Glaucoma": GLAUCOMA_GWAS_VARIANTS,
}


# ---------------------------------------------------------------------------
# Environmental profile
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentalProfile:
    """
    Individual lifestyle and environmental covariates for GxE modeling.

    All continuous variables should be provided on their natural scale;
    normalization is handled internally.

    Attributes
    ----------
    age           : int   — current age in years
    smoking       : bool  — current or former smoker
    bmi           : float — body mass index (kg/m²)
    uv_exposure   : float — cumulative UV exposure score [0, 1]
    diabetes      : bool  — diagnosed with type 2 diabetes
    systolic_bp   : float — systolic blood pressure (mmHg)
    diet_quality  : float — diet quality score [0, 1] (1 = excellent)
    physical_act  : float — physical activity level [0, 1]
    """
    age: int = 50
    smoking: bool = False
    bmi: float = 24.0
    uv_exposure: float = 0.3
    diabetes: bool = False
    systolic_bp: float = 120.0
    diet_quality: float = 0.6
    physical_act: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Convert to a normalized feature vector for GxE computation."""
        return np.array([
            (self.age - 40) / 40.0,           # centered at 40, scaled
            float(self.smoking),
            (self.bmi - 18.5) / 20.0,          # excess above healthy
            self.uv_exposure,
            float(self.diabetes),
            (self.systolic_bp - 80.0) / 80.0,  # scaled
            1.0 - self.diet_quality,            # invert: higher = worse
            1.0 - self.physical_act,            # invert: higher = worse
        ])


# ---------------------------------------------------------------------------
# GxE interaction model
# ---------------------------------------------------------------------------

@dataclass
class GxEModel:
    """
    Gene-environment interaction model.

    Models how environmental/lifestyle factors modulate the genetic PRS.
    The composite risk is:

        log_odds = PRS + β_env · E + γ · (PRS × E)
                                       ↑
                               interaction term

    where E is the environmental feature vector.

    Parameters
    ----------
    env_weights : np.ndarray
        Coefficients (β_env) for direct environmental effects.
        Shape: (n_env_features,)
    interaction_weights : np.ndarray
        Coefficients (γ) for GxE interaction terms (PRS × each env feature).
        Shape: (n_env_features,)
    """
    env_weights: np.ndarray = field(
        default_factory=lambda: np.array([
            0.08,   # age
            0.25,   # smoking — strong independent risk for AMD
            0.10,   # BMI
            0.15,   # UV exposure
            0.20,   # diabetes
            0.08,   # blood pressure
            0.12,   # poor diet
            0.07,   # physical inactivity
        ])
    )
    interaction_weights: np.ndarray = field(
        default_factory=lambda: np.array([
            0.05,   # age × PRS
            0.15,   # smoking × PRS — amplifies genetic risk
            0.04,   # BMI × PRS
            0.08,   # UV × PRS
            0.10,   # diabetes × PRS
            0.03,   # BP × PRS
            0.06,   # diet × PRS
            0.03,   # activity × PRS
        ])
    )

    def compute(self, prs: float, env_profile: EnvironmentalProfile) -> float:
        """
        Compute the GxE-adjusted log-odds risk.

        Returns
        -------
        float — adjusted log-odds (not yet normalized to [0,1])
        """
        E = env_profile.to_vector()
        direct_env = self.env_weights @ E
        interaction = self.interaction_weights @ (prs * E)
        return prs + direct_env + interaction


# ---------------------------------------------------------------------------
# PRS Model
# ---------------------------------------------------------------------------

class PRSModel:
    """
    Polygenic Risk Score model for eye disease.

    Computes a weighted sum of risk allele dosages across GWAS-identified
    variants, then applies GxE adjustment to produce a final risk score.

    Parameters
    ----------
    disease : str — "AMD" or "Glaucoma"
    """

    def __init__(self, disease: str = "AMD"):
        if disease not in DISEASE_VARIANTS:
            raise ValueError(f"Disease '{disease}' not recognized. Options: {list(DISEASE_VARIANTS.keys())}")

        self.disease = disease
        self.variants = DISEASE_VARIANTS[disease].copy()
        self.n_variants = len(self.variants)
        self.gxe = GxEModel()

        # Precompute mean-centering term for PRS standardization
        # E[PRS] = Σ 2 * p_i * β_i  (assuming Hardy-Weinberg, additive model)
        self._prs_mean = 2.0 * (
            self.variants["risk_allele_freq"] * self.variants["effect_size"]
        ).sum()

        # Var[PRS] = Σ 2 * p_i * (1 - p_i) * β_i²
        self._prs_std = np.sqrt(
            2.0 * (
                self.variants["risk_allele_freq"]
                * (1 - self.variants["risk_allele_freq"])
                * self.variants["effect_size"] ** 2
            ).sum()
        )

    # ------------------------------------------------------------------
    # Genotype simulation (for testing without real data)
    # ------------------------------------------------------------------

    def simulate_genotype(
        self, n_individuals: int = 1, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate genotype dosage matrix under Hardy-Weinberg equilibrium.

        Each individual × variant entry is drawn from Binomial(2, p)
        where p is the risk allele frequency.

        Returns
        -------
        np.ndarray of shape (n_individuals, n_variants)
            Dosage values in {0, 1, 2}.
        """
        if seed is not None:
            np.random.seed(seed)

        freqs = self.variants["risk_allele_freq"].values
        dosages = np.column_stack([
            np.random.binomial(2, p, size=n_individuals) for p in freqs
        ])
        return dosages

    # ------------------------------------------------------------------
    # PRS computation
    # ------------------------------------------------------------------

    def compute_prs(self, genotype: np.ndarray) -> np.ndarray:
        """
        Compute raw and standardized PRS for one or more individuals.

        PRS_i = Σ_j β_j × dosage_{ij}

        Standardized: z = (PRS - mean) / std  (under HWE)

        Parameters
        ----------
        genotype : np.ndarray of shape (n_variants,) or (n_individuals, n_variants)
            Dosage values {0, 1, 2} per variant.

        Returns
        -------
        np.ndarray — standardized PRS per individual (z-scores)
        """
        genotype = np.atleast_2d(genotype)
        effects = self.variants["effect_size"].values
        raw_prs = genotype @ effects  # (n_individuals,)
        return (raw_prs - self._prs_mean) / (self._prs_std + 1e-10)

    def prs_to_probability(self, prs_z: np.ndarray) -> np.ndarray:
        """
        Convert standardized PRS z-scores to probabilities via sigmoid.

        A z-score of 0 maps to ~0.5 baseline. Rescale so output is [0, 1]
        and a z-score of 0 maps to ~0.3 (population baseline).
        """
        # Shift so z=0 → ~0.3 probability
        return expit(prs_z * 0.8 - 0.5)

    # ------------------------------------------------------------------
    # Full risk computation with GxE
    # ------------------------------------------------------------------

    def compute_risk(
        self,
        genotype: np.ndarray,
        env_profile: Optional[EnvironmentalProfile] = None,
    ) -> float:
        """
        Compute a composite genetic + environmental risk score in [0, 1].

        Parameters
        ----------
        genotype    : np.ndarray of shape (n_variants,) — dosage values
        env_profile : EnvironmentalProfile or None (uses defaults if None)

        Returns
        -------
        float in [0, 1] — composite risk score
        """
        if env_profile is None:
            env_profile = EnvironmentalProfile()

        prs_z = self.compute_prs(genotype)[0]
        prs_prob = float(self.prs_to_probability(np.array([prs_z]))[0])

        # GxE adjustment
        log_odds_adjusted = self.gxe.compute(prs_z, env_profile)
        composite_prob = float(expit(log_odds_adjusted * 0.6 - 0.2))

        return float(np.clip(composite_prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def compute_population_risk(
        self,
        n: int = 1000,
        env_profile: Optional[EnvironmentalProfile] = None,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Simulate risk scores for a synthetic population of n individuals.
        Useful for visualizing the population risk distribution.
        """
        genotypes = self.simulate_genotype(n_individuals=n, seed=seed)
        if env_profile is None:
            env_profile = EnvironmentalProfile()

        risks = np.array([
            self.compute_risk(genotypes[i], env_profile)
            for i in range(n)
        ])
        return risks

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_population_distribution(
        self,
        n: int = 2000,
        env_profile: Optional[EnvironmentalProfile] = None,
        highlight_score: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the population distribution of risk scores, optionally
        highlighting a specific individual's score.
        """
        risks = self.compute_population_risk(n=n, env_profile=env_profile)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        ax.hist(risks, bins=50, color="#3498db", alpha=0.7, edgecolor="#0f0f0f", linewidth=0.5)

        if highlight_score is not None:
            ax.axvline(highlight_score, color="#e74c3c", linewidth=2.5,
                       label=f"Your score: {highlight_score:.3f}")
            pct = float((risks < highlight_score).mean() * 100)
            ax.text(highlight_score + 0.01, ax.get_ylim()[1] * 0.85,
                    f"{pct:.0f}th percentile",
                    color="#e74c3c", fontsize=10)
            ax.legend(labelcolor="white", facecolor="#1a1a1a", edgecolor="#444")

        ax.set_xlabel("Composite Risk Score", color="white", fontsize=12)
        ax.set_ylabel("Count", color="white", fontsize=12)
        ax.set_title(
            f"{self.disease} — Population Risk Score Distribution (n={n:,})",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def plot_gxe_sensitivity(
        self,
        genotype: np.ndarray,
        env_variable: str = "smoking",
        n_points: int = 50,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot how a single environmental variable modulates the composite risk
        for a given genotype — visualizing the GxE interaction.

        Parameters
        ----------
        genotype     : individual genotype vector
        env_variable : one of the EnvironmentalProfile continuous fields
        """
        continuous_vars = ["bmi", "uv_exposure", "systolic_bp", "diet_quality", "physical_act"]
        binary_vars = ["smoking", "diabetes"]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f0f0f")
        ax.set_facecolor("#0f0f0f")

        if env_variable in continuous_vars:
            ranges = {
                "bmi": (18.0, 40.0),
                "uv_exposure": (0.0, 1.0),
                "systolic_bp": (90.0, 180.0),
                "diet_quality": (0.0, 1.0),
                "physical_act": (0.0, 1.0),
            }
            lo, hi = ranges.get(env_variable, (0.0, 1.0))
            x_vals = np.linspace(lo, hi, n_points)
            risks = []
            for val in x_vals:
                env = EnvironmentalProfile(**{env_variable: val})
                risks.append(self.compute_risk(genotype, env))

            ax.plot(x_vals, risks, color="#e67e22", linewidth=2.5)
            ax.set_xlabel(env_variable.replace("_", " ").title(), color="white", fontsize=12)
            ax.set_title(
                f"GxE Sensitivity: {env_variable} vs. {self.disease} Risk",
                color="white", fontsize=13, fontweight="bold"
            )

        elif env_variable in binary_vars:
            env_off = EnvironmentalProfile(**{env_variable: False})
            env_on  = EnvironmentalProfile(**{env_variable: True})
            r_off = self.compute_risk(genotype, env_off)
            r_on  = self.compute_risk(genotype, env_on)

            labels = [f"{env_variable}=False", f"{env_variable}=True"]
            colors = ["#2ecc71", "#e74c3c"]
            bars = ax.bar(labels, [r_off, r_on], color=colors, width=0.4)
            ax.set_ylabel("Composite Risk Score", color="white", fontsize=12)
            ax.set_title(
                f"GxE Effect: {env_variable} on {self.disease} Risk",
                color="white", fontsize=13, fontweight="bold"
            )
            for bar, val in zip(bars, [r_off, r_on]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.3f}", ha="center", color="white", fontsize=11)
        else:
            raise ValueError(f"env_variable '{env_variable}' not recognized.")

        ax.set_ylim(0, 1)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def print_summary(self, genotype: np.ndarray, env_profile: EnvironmentalProfile) -> None:
        """Print a readable risk summary for one individual."""
        prs_z = float(self.compute_prs(genotype)[0])
        prs_prob = float(self.prs_to_probability(np.array([prs_z]))[0])
        composite = self.compute_risk(genotype, env_profile)

        print(f"\n{'='*50}")
        print(f"  {self.disease} — Individual Risk Summary")
        print(f"{'='*50}")
        print(f"  Raw PRS (z-score)    : {prs_z:+.3f}")
        print(f"  PRS probability      : {prs_prob:.3f}")
        print(f"  Composite risk (GxE) : {composite:.3f}")
        print(f"\n  Environmental inputs:")
        for key, val in env_profile.__dict__.items():
            print(f"    {key:<20}: {val}")
        print(f"{'='*50}\n")