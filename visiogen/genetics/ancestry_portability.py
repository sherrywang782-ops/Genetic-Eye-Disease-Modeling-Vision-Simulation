"""
ancestry_portability.py
=======================
Ancestry Portability Analysis for Polygenic Risk Scores (PRS).

PRS models trained on European-ancestry GWAS data perform less accurately
when applied to individuals of other ancestries. This module quantifies
that degradation mathematically.

Key concepts:
    - Fst (Fixation Index): measures genetic distance between populations.
      Fst = 0 means identical allele frequencies; Fst = 1 means completely
      diverged. It is computed per-variant and averaged across the genome.
    - LD (Linkage Disequilibrium): correlation structure between variants.
      Differs across ancestries, causing effect size estimates to
      mis-calibrate when transferred.
    - PRS portability: prediction accuracy (measured by R² or AUC) as a
      function of genetic distance from the training population.

This module:
    1. Simulates ancestry-specific genotype populations
    2. Computes PRS using European-trained weights on each population
    3. Measures prediction accuracy across ancestries
    4. Visualizes the degradation curve
    5. Demonstrates the improvement from multi-ancestry GWAS re-weighting

Reference:
    Martin et al. (2019) Nature Genetics — "Clinical use of current
    polygenic risk scores may exacerbate health disparities"

Author: VisioGen Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Ancestry definitions
# ---------------------------------------------------------------------------

# Approximate mean Fst relative to European populations
# Source: 1000 Genomes Project population genetics literature
ANCESTRIES = {
    "European":          {"fst": 0.00, "color": "#3498db", "label": "European (training)"},
    "Latino":            {"fst": 0.04, "color": "#2ecc71", "label": "Latino/Admixed American"},
    "South Asian":       {"fst": 0.08, "color": "#f1c40f", "label": "South Asian"},
    "East Asian":        {"fst": 0.11, "color": "#e67e22", "label": "East Asian"},
    "African":           {"fst": 0.15, "color": "#e74c3c", "label": "African"},
}

# AMD risk allele frequencies vary across ancestries.
# These are illustrative estimates derived from population genetics literature.
# European frequencies are the reference (used in GWAS training).
# Other ancestries have frequencies shifted proportional to Fst.
AMD_FREQ_MODIFIERS = {
    # rsid          EUR    Latino  S.Asian  E.Asian  African
    "rs1061170":   [0.35,  0.28,   0.22,    0.10,    0.08],
    "rs10490924":  [0.22,  0.18,   0.15,    0.12,    0.05],
    "rs2230199":   [0.23,  0.21,   0.20,    0.18,    0.16],
    "rs9332739":   [0.12,  0.11,   0.10,    0.08,    0.07],
    "rs10737680":  [0.55,  0.50,   0.48,    0.42,    0.38],
    "rs3750847":   [0.25,  0.22,   0.20,    0.15,    0.12],
    "rs1329428":   [0.60,  0.56,   0.52,    0.45,    0.40],
    "rs4698775":   [0.38,  0.35,   0.32,    0.28,    0.25],
    "rs13081855":  [0.14,  0.13,   0.12,    0.10,    0.09],
    "rs8135665":   [0.45,  0.42,   0.40,    0.36,    0.32],
    "rs3812111":   [0.50,  0.47,   0.44,    0.40,    0.36],
    "rs429608":    [0.18,  0.17,   0.16,    0.14,    0.12],
    "rs11200638":  [0.20,  0.18,   0.16,    0.12,    0.08],
    "rs2736911":   [0.40,  0.38,   0.36,    0.32,    0.28],
    "rs1864163":   [0.15,  0.14,   0.13,    0.11,    0.10],
}

ANCESTRY_ORDER = list(ANCESTRIES.keys())
FREQ_MATRIX = np.array([AMD_FREQ_MODIFIERS[rsid] for rsid in AMD_FREQ_MODIFIERS])
# Shape: (n_variants, n_ancestries)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortabilityResult:
    """
    Results of a PRS portability analysis across ancestries.

    Attributes
    ----------
    ancestry_names  : list of ancestry labels
    fst_values      : Fst (genetic distance from European training population)
    r2_values       : PRS prediction R² per ancestry
    auc_values      : AUC per ancestry
    r2_ci           : (lower, upper) confidence intervals on R²
    auc_ci          : (lower, upper) confidence intervals on AUC
    """
    ancestry_names: list
    fst_values: np.ndarray
    r2_values: np.ndarray
    auc_values: np.ndarray
    r2_ci: tuple
    auc_ci: tuple


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

class AncestryPortabilityAnalyzer:
    """
    Quantifies PRS prediction accuracy degradation across ancestries.

    Parameters
    ----------
    effect_sizes : np.ndarray
        GWAS effect sizes (log-odds ratios) from European-ancestry training.
        Shape: (n_variants,)
    n_individuals : int
        Number of individuals to simulate per ancestry.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        effect_sizes: np.ndarray,
        n_individuals: int = 2000,
        seed: int = 42,
    ):
        self.effect_sizes = effect_sizes
        self.n_variants = len(effect_sizes)
        self.n_individuals = n_individuals
        self.seed = seed

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_population(
        self,
        allele_freqs: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate genotype dosage matrix for a population.

        Each individual × variant entry ~ Binomial(2, p_j).

        Returns
        -------
        np.ndarray of shape (n, n_variants) — dosage values in {0, 1, 2}
        """
        return np.column_stack([
            rng.binomial(2, p, size=n) for p in allele_freqs
        ])

    def _simulate_true_risk(
        self,
        genotypes: np.ndarray,
        true_effect_sizes: np.ndarray,
        prevalence: float = 0.08,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate binary disease labels using the true genetic architecture.

        The liability threshold model:
            liability_i = Σ_j β_j * g_ij + ε_i,  ε ~ N(0, 1)
            disease_i = 1 if liability_i > threshold

        The threshold is set to match the target prevalence.

        Parameters
        ----------
        genotypes        : (n, n_variants) dosage matrix
        true_effect_sizes: true causal effect sizes for this ancestry
        prevalence       : population disease prevalence
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        liability = genotypes @ true_effect_sizes
        liability += rng.normal(0, liability.std() * 0.5, size=len(liability))

        # Threshold for target prevalence
        threshold = np.quantile(liability, 1 - prevalence)
        return (liability > threshold).astype(int)

    def _compute_prs(self, genotypes: np.ndarray) -> np.ndarray:
        """
        Compute PRS using European-trained effect sizes.
        Returns standardized z-scores.
        """
        raw = genotypes @ self.effect_sizes
        return (raw - raw.mean()) / (raw.std() + 1e-10)

    # ------------------------------------------------------------------
    # Fst computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fst(freq_pop1: np.ndarray, freq_pop2: np.ndarray) -> float:
        """
        Compute mean Fst (Weir & Cockerham 1984) between two populations.

        Fst_j = (p̄_j(1 - p̄_j) - [p1_j(1-p1_j) + p2_j(1-p2_j)]/2) /
                 p̄_j(1 - p̄_j)

        where p̄_j = (p1_j + p2_j) / 2

        Returns the mean Fst across all variants.
        """
        p_bar = (freq_pop1 + freq_pop2) / 2
        numerator   = p_bar * (1 - p_bar) - (
            freq_pop1 * (1 - freq_pop1) + freq_pop2 * (1 - freq_pop2)
        ) / 2
        denominator = p_bar * (1 - p_bar) + 1e-10
        fst_per_variant = np.clip(numerator / denominator, 0, 1)
        return float(fst_per_variant.mean())

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        prs: np.ndarray,
        labels: np.ndarray,
        metric: str = "r2",
        n_bootstrap: int = 200,
        ci: float = 0.95,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple:
        """
        Bootstrap confidence interval for R² or AUC.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        n = len(prs)
        scores = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            p, l = prs[idx], labels[idx]
            if metric == "r2":
                r, _ = pearsonr(p, l)
                scores.append(r ** 2)
            else:
                if len(np.unique(l)) > 1:
                    scores.append(roc_auc_score(l, p))

        alpha = (1 - ci) / 2
        return (
            float(np.quantile(scores, alpha)),
            float(np.quantile(scores, 1 - alpha)),
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def run(
        self,
        n_bootstrap: int = 200,
        ld_decay: float = 0.3,
    ) -> PortabilityResult:
        """
        Run the full portability analysis across all ancestries.

        Parameters
        ----------
        n_bootstrap : int — number of bootstrap iterations for CIs
        ld_decay    : float — controls how much LD mismatch degrades
                      effect size estimates across ancestries (0 = no decay)

        Returns
        -------
        PortabilityResult
        """
        rng = np.random.default_rng(self.seed)
        eur_freqs = FREQ_MATRIX[:, 0]  # European reference frequencies

        r2_values, auc_values = [], []
        r2_lower, r2_upper   = [], []
        auc_lower, auc_upper = [], []
        fst_values = []

        for anc_idx, ancestry in enumerate(ANCESTRY_ORDER):
            anc_freqs = FREQ_MATRIX[:, anc_idx]

            # Compute Fst from European reference
            fst = self.compute_fst(eur_freqs, anc_freqs)
            fst_values.append(fst)

            # Simulate genotypes for this ancestry
            genotypes = self._simulate_population(anc_freqs, self.n_individuals, rng)

            # True effect sizes for this ancestry are attenuated by LD mismatch.
            # The attenuation scales with Fst — higher Fst = more LD mismatch.
            # This is a simplified model of the LD decay phenomenon.
            attenuation = np.exp(-ld_decay * fst * 10)
            true_effects = self.effect_sizes * attenuation

            # Simulate disease labels using ancestry-appropriate true effects
            labels = self._simulate_true_risk(genotypes, true_effects, rng=rng)

            # Compute PRS using European-trained weights (portability test)
            prs = self._compute_prs(genotypes)

            # Prediction accuracy
            r, _ = pearsonr(prs, labels)
            r2 = r ** 2
            r2_values.append(r2)

            if len(np.unique(labels)) > 1:
                auc = roc_auc_score(labels, prs)
            else:
                auc = 0.5
            auc_values.append(auc)

            # Bootstrap CIs
            r2_lo, r2_hi = self._bootstrap_ci(prs, labels, "r2", n_bootstrap, rng=rng)
            r2_lower.append(r2_lo); r2_upper.append(r2_hi)

            auc_lo, auc_hi = self._bootstrap_ci(prs, labels, "auc", n_bootstrap, rng=rng)
            auc_lower.append(auc_lo); auc_upper.append(auc_hi)

        return PortabilityResult(
            ancestry_names=ANCESTRY_ORDER,
            fst_values=np.array(fst_values),
            r2_values=np.array(r2_values),
            auc_values=np.array(auc_values),
            r2_ci=(np.array(r2_lower), np.array(r2_upper)),
            auc_ci=(np.array(auc_lower), np.array(auc_upper)),
        )

    # ------------------------------------------------------------------
    # Multi-ancestry re-weighting (improvement simulation)
    # ------------------------------------------------------------------

    def run_multi_ancestry(self, n_bootstrap: int = 200) -> PortabilityResult:
        """
        Simulate the improvement in portability from multi-ancestry
        GWAS re-weighting.

        In a multi-ancestry GWAS, effect sizes are estimated jointly
        across populations, reducing the LD mismatch penalty. We model
        this as a reduced ld_decay parameter.
        """
        # Multi-ancestry training reduces LD decay effect by ~60%
        return self.run(n_bootstrap=n_bootstrap, ld_decay=0.12)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_degradation_curve(
        self,
        result: PortabilityResult,
        result_multi: Optional[PortabilityResult] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot PRS prediction accuracy (R²) vs. genetic distance (Fst)
        from the European training population.

        Optionally overlays the multi-ancestry re-weighted result.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0f0f0f")
        fig.suptitle(
            "PRS Portability — Prediction Accuracy vs. Genetic Distance",
            color="white", fontsize=14, fontweight="bold"
        )

        metrics = [
            ("r2_values",  "r2_ci",  "R² (Variance Explained)", "r2"),
            ("auc_values", "auc_ci", "AUC (Discrimination)",     "auc"),
        ]

        for ax, (val_attr, ci_attr, ylabel, _) in zip(axes, metrics):
            ax.set_facecolor("#0f0f0f")

            fst  = result.fst_values
            vals = getattr(result, val_attr)
            lo, hi = getattr(result, ci_attr)

            colors = [ANCESTRIES[a]["color"] for a in result.ancestry_names]

            # CI band — European-only trained
            ax.fill_between(fst, lo, hi, color="#3498db", alpha=0.12, label="_nolegend_")
            ax.plot(fst, vals, color="#3498db", linewidth=2.5,
                    label="European-trained PRS", zorder=3)

            # Scatter points per ancestry
            for i, (anc, color) in enumerate(zip(result.ancestry_names, colors)):
                ax.scatter(fst[i], vals[i], color=color, s=100, zorder=5,
                           edgecolors="white", linewidths=0.8)
                ax.annotate(
                    ANCESTRIES[anc]["label"],
                    xy=(fst[i], vals[i]),
                    xytext=(fst[i] + 0.003, vals[i] + 0.008),
                    color=color, fontsize=8.5,
                )

            # Multi-ancestry overlay
            if result_multi is not None:
                vals_m = getattr(result_multi, val_attr)
                lo_m, hi_m = getattr(result_multi, ci_attr)
                ax.fill_between(fst, lo_m, hi_m, color="#2ecc71", alpha=0.12)
                ax.plot(fst, vals_m, color="#2ecc71", linewidth=2.5,
                        linestyle="--", label="Multi-ancestry PRS", zorder=3)

            ax.set_xlabel("Fst (Genetic Distance from European)", color="white", fontsize=11)
            ax.set_ylabel(ylabel, color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(labelcolor="white", facecolor="#1a1a1a",
                      edgecolor="#444", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def plot_ancestry_comparison(
        self,
        result: PortabilityResult,
        result_multi: Optional[PortabilityResult] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Bar chart comparing R² and AUC across ancestries,
        with and without multi-ancestry re-weighting.
        """
        n = len(result.ancestry_names)
        x = np.arange(n)
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f0f0f")
        fig.suptitle(
            "PRS Performance Across Ancestries",
            color="white", fontsize=14, fontweight="bold"
        )

        metrics = [
            ("r2_values",  "r2_ci",  "R² (Variance Explained)"),
            ("auc_values", "auc_ci", "AUC"),
        ]

        for ax, (val_attr, ci_attr, ylabel) in zip(axes, metrics):
            ax.set_facecolor("#0f0f0f")

            vals = getattr(result, val_attr)
            lo, hi = getattr(result, ci_attr)
            colors = [ANCESTRIES[a]["color"] for a in result.ancestry_names]
            yerr = np.array([vals - lo, hi - vals])

            if result_multi is not None:
                bars1 = ax.bar(x - width/2, vals, width, color=colors,
                               alpha=0.75, label="EUR-trained PRS", yerr=yerr,
                               error_kw={"ecolor": "white", "capsize": 3})
                vals_m = getattr(result_multi, val_attr)
                lo_m, hi_m = getattr(result_multi, ci_attr)
                yerr_m = np.array([vals_m - lo_m, hi_m - vals_m])
                bars2 = ax.bar(x + width/2, vals_m, width, color=colors,
                               alpha=1.0, label="Multi-ancestry PRS", yerr=yerr_m,
                               error_kw={"ecolor": "white", "capsize": 3},
                               edgecolor="white", linewidth=0.8)
            else:
                ax.bar(x, vals, color=colors, alpha=0.85, yerr=yerr,
                       error_kw={"ecolor": "white", "capsize": 3})

            ax.set_xticks(x)
            ax.set_xticklabels(result.ancestry_names, color="white", fontsize=9, rotation=15)
            ax.set_ylabel(ylabel, color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if result_multi is not None:
                ax.legend(labelcolor="white", facecolor="#1a1a1a",
                          edgecolor="#444", fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def print_summary(
        self,
        result: PortabilityResult,
        result_multi: Optional[PortabilityResult] = None,
    ) -> None:
        """Print a readable summary table of portability results."""
        print(f"\n{'='*75}")
        print(f"  PRS Portability Analysis — AMD")
        print(f"{'='*75}")

        if result_multi:
            print(f"  {'Ancestry':<20} {'Fst':>6} {'R² (EUR)':>10} {'R² (Multi)':>12} {'ΔAUC':>8}")
        else:
            print(f"  {'Ancestry':<20} {'Fst':>6} {'R²':>10} {'AUC':>10}")
        print(f"  {'-'*65}")

        eur_r2  = result.r2_values[0]
        eur_auc = result.auc_values[0]

        for i, anc in enumerate(result.ancestry_names):
            fst  = result.fst_values[i]
            r2   = result.r2_values[i]
            auc  = result.auc_values[i]

            if result_multi:
                r2_m  = result_multi.r2_values[i]
                d_auc = result_multi.auc_values[i] - auc
                pct_retained = r2 / eur_r2 * 100
                print(f"  {anc:<20} {fst:>6.3f} {r2:>10.4f} {r2_m:>12.4f} {d_auc:>+8.4f}")
            else:
                pct_retained = r2 / eur_r2 * 100
                print(f"  {anc:<20} {fst:>6.3f} {r2:>10.4f} {auc:>10.4f}  ({pct_retained:.0f}% of EUR R²)")

        print(f"{'='*75}\n")
        print(f"  Key finding: R² drops from {eur_r2:.4f} in Europeans to")
        print(f"  {result.r2_values[-1]:.4f} in Africans — a {(1 - result.r2_values[-1]/eur_r2)*100:.0f}% reduction.")
        if result_multi:
            eur_r2_m = result_multi.r2_values[0]
            afr_r2_m = result_multi.r2_values[-1]
            print(f"\n  With multi-ancestry re-weighting: R² improves from")
            print(f"  {result.r2_values[-1]:.4f} to {afr_r2_m:.4f} in Africans (+{(afr_r2_m - result.r2_values[-1])*100:.1f}%).")
        print()