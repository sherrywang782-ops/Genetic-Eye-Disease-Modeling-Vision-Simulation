"""
disease_filters.py
==================
Visual simulation pipeline for eye disease perception.

Transforms normal images to simulate the visual experience of
individuals at different stages of various eye diseases.

Each disease's simulation is parameterized by a severity score in [0, 1],
which is directly fed from the Bayesian Markov progression model output.

Supported diseases:
    - AMD (Age-related Macular Degeneration) — central scotoma
    - Glaucoma                               — peripheral field loss
    - Cataracts                              — diffuse haze, contrast loss
    - Diabetic Retinopathy                   — floaters, hemorrhages, blur
    - Retinitis Pigmentosa                   — ring scotoma → tunnel vision
    - Color Blindness (Deuteranopia)         — red-green color remapping

Usage:
    from disease_filters import EyeDiseaseSimulator
    sim = EyeDiseaseSimulator()
    result = sim.simulate(image, disease="AMD", severity=0.6)
    sim.show_comparison(image, disease="AMD", severity=0.6)

Author: VisioGen Project
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Union
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Severity stage labels
# ---------------------------------------------------------------------------

DISEASE_STAGES = {
    "AMD":                  ["Healthy", "Early", "Intermediate", "Advanced", "Severe"],
    "Glaucoma":             ["Healthy", "Suspected", "Early", "Moderate", "Advanced", "Severe"],
    "Cataracts":            ["Healthy", "Mild", "Moderate", "Dense", "Severe"],
    "Diabetic Retinopathy": ["Healthy", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"],
    "Retinitis Pigmentosa": ["Healthy", "Early", "Moderate", "Advanced", "Tunnel", "Severe"],
    "Color Blindness":      ["Normal", "Mild", "Moderate", "Strong", "Complete"],
}


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

def _apply_amd(image: np.ndarray, severity: float) -> np.ndarray:
    """
    AMD: Progressive central scotoma (dark/blurry patch at fixation point).
    At low severity: subtle central blur.
    At high severity: large dark central region with distortion (metamorphopsia).
    """
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # Scotoma radius scales with severity: 0 → 0%, 1 → 35% of image width
    scotoma_radius = int(severity * 0.35 * min(h, w))

    if scotoma_radius < 2:
        return out.astype(np.uint8)

    # Create a blurred + darkened version for the scotoma region
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=severity * 25 + 1)
    darkened = blurred * (1.0 - severity * 0.85)

    # Smooth circular mask: 1 inside scotoma, 0 outside
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip((scotoma_radius - dist) / max(scotoma_radius * 0.3, 1), 0, 1)
    mask = mask[:, :, np.newaxis]

    out = out * (1 - mask) + darkened * mask

    # Add mild peripheral blur for severe AMD
    if severity > 0.5:
        peripheral_blur = cv2.GaussianBlur(out, (0, 0), sigmaX=(severity - 0.5) * 10)
        edge_mask = np.clip((dist - scotoma_radius * 1.5) / (min(h, w) * 0.3), 0, 1)
        edge_mask = edge_mask[:, :, np.newaxis]
        out = out * (1 - edge_mask * 0.3) + peripheral_blur * edge_mask * 0.3

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_glaucoma(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Glaucoma: Progressive peripheral field loss (tunnel vision).
    The remaining visual field contracts toward the center.
    """
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # Field radius: starts large (full image), contracts with severity
    field_radius = (1.0 - severity * 0.85) * min(h, w) * 0.55
    field_radius = max(field_radius, min(h, w) * 0.05)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Soft edge: blend into black
    transition_width = field_radius * 0.35
    mask = np.clip((field_radius - dist) / transition_width, 0, 1)
    mask = mask[:, :, np.newaxis]

    out = out * mask  # black outside field

    # Slight blur at edges of remaining field
    if severity > 0.3:
        edge_blur = cv2.GaussianBlur(out, (0, 0), sigmaX=severity * 8)
        edge_mask = np.clip((dist - field_radius * 0.7) / (field_radius * 0.4), 0, 1)
        edge_mask = np.clip(edge_mask * mask[:, :, 0][:, :, np.newaxis], 0, 1)
        out = out * (1 - edge_mask * 0.6) + edge_blur * edge_mask * 0.6

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_cataracts(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Cataracts: Diffuse haze, contrast reduction, yellowing, halos around lights.
    """
    out = image.copy().astype(np.float32)

    # Blur (haze increases with severity)
    if severity > 0.05:
        sigma = severity * 12
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)

    # Contrast reduction
    contrast_factor = 1.0 - severity * 0.55
    out = (out - 127.5) * contrast_factor + 127.5

    # Brightness increase (cataracts cause glare/washed-out appearance)
    out = out + severity * 30

    # Yellow-brown tint (lens yellowing)
    if len(out.shape) == 3 and out.shape[2] == 3:
        yellow_tint = np.array([1.0, 0.95, 1.0 - severity * 0.30])  # reduce blue
        out = out * yellow_tint

    # Halo effect: bright regions bloom outward
    if severity > 0.4:
        bright_mask = (out.mean(axis=2) > 180).astype(np.float32)
        halo = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=(severity - 0.4) * 30)
        halo = halo[:, :, np.newaxis] * np.array([255, 245, 200])
        out = out + halo * (severity - 0.4) * 0.4

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_diabetic_retinopathy(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Diabetic Retinopathy: Blurring, floaters (dark spots), and at severe
    stages, vitreous hemorrhage (dark streaks/clouds).
    """
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]

    # General blur
    if severity > 0.1:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=severity * 8)

    # Contrast reduction
    out = (out - 127.5) * (1.0 - severity * 0.3) + 127.5

    # Add dark floaters (number and size scale with severity)
    n_floaters = int(severity * 20)
    rng = np.random.default_rng(seed=42)
    for _ in range(n_floaters):
        fx = rng.integers(0, w)
        fy = rng.integers(0, h)
        fr = rng.integers(3, int(severity * 25) + 5)
        darkness = rng.uniform(0.4, 0.85)
        cv2.circle(out, (fx, fy), fr, (0, 0, 0), -1)
        # Soft edge
        floater_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(floater_mask, (fx, fy), fr, 1.0, -1)
        floater_mask = cv2.GaussianBlur(floater_mask, (0, 0), sigmaX=fr * 0.5)
        out = out * (1 - floater_mask[:, :, np.newaxis] * darkness)

    # Hemorrhage streaks at high severity
    if severity > 0.65:
        n_hemorrhages = int((severity - 0.65) * 8)
        for _ in range(n_hemorrhages):
            x1 = rng.integers(0, w)
            y1 = rng.integers(0, h)
            x2 = x1 + rng.integers(-60, 60)
            y2 = y1 + rng.integers(-60, 60)
            thickness = rng.integers(3, 10)
            cv2.line(out, (x1, y1), (x2, y2), (20, 0, 0), thickness)

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_retinitis_pigmentosa(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Retinitis Pigmentosa: Ring scotoma that progresses inward from periphery,
    leaving only a central tunnel of vision.
    """
    out = image.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # RP erodes from outside in; at full severity only a small tunnel remains
    preserved_radius = (1.0 - severity * 0.92) * max_dist
    preserved_radius = max(preserved_radius, max_dist * 0.04)

    transition = preserved_radius * 0.25
    mask = np.clip((preserved_radius - dist) / transition, 0, 1)
    mask = mask[:, :, np.newaxis]

    # Night blindness: desaturate and dim peripheral regions
    gray = out.mean(axis=2, keepdims=True)
    desaturated = out * 0.2 + gray * 0.8

    # Peripheral region: dark + desaturated
    out = out * mask + desaturated * (1 - mask) * (1 - severity * 0.95)

    # Scatter/noise typical of RP
    if severity > 0.2:
        noise = np.random.normal(0, severity * 8, out.shape)
        peripheral_noise = noise * (1 - mask)
        out = np.clip(out + peripheral_noise, 0, 255)

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_color_blindness(image: np.ndarray, severity: float) -> np.ndarray:
    """
    Deuteranopia (red-green color blindness): Linear color space transform.
    Based on Brettel et al. (1997) simulation matrices.
    Severity interpolates between normal and full deuteranopia.
    """
    out = image.copy().astype(np.float32) / 255.0

    # Deuteranopia simulation matrix (sRGB linear approximation)
    # Maps normal trichromat RGB to deuteranope RGB
    deuteranopia_matrix = np.array([
        [0.625, 0.375, 0.000],
        [0.700, 0.300, 0.000],
        [0.000, 0.300, 0.700],
    ])
    identity = np.eye(3)
    transform = (1 - severity) * identity + severity * deuteranopia_matrix

    if len(out.shape) == 3 and out.shape[2] == 3:
        h, w = out.shape[:2]
        flat = out.reshape(-1, 3)
        transformed = flat @ transform.T
        out = transformed.reshape(h, w, 3)

    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------

FILTER_MAP = {
    "AMD":                  _apply_amd,
    "Glaucoma":             _apply_glaucoma,
    "Cataracts":            _apply_cataracts,
    "Diabetic Retinopathy": _apply_diabetic_retinopathy,
    "Retinitis Pigmentosa": _apply_retinitis_pigmentosa,
    "Color Blindness":      _apply_color_blindness,
}


class EyeDiseaseSimulator:
    """
    Visual simulation pipeline for eye disease perception.

    Parameters
    ----------
    None

    Examples
    --------
    >>> sim = EyeDiseaseSimulator()
    >>> result = sim.simulate(image, disease="AMD", severity=0.6)
    >>> sim.show_comparison(image, disease="AMD", severity=0.6)
    >>> sim.show_progression(image, disease="Glaucoma", n_steps=5)
    """

    SUPPORTED = list(FILTER_MAP.keys())

    def simulate(
        self,
        image: Union[np.ndarray, str, Path],
        disease: str,
        severity: float,
    ) -> np.ndarray:
        """
        Apply a disease simulation filter to an image.

        Parameters
        ----------
        image    : np.ndarray (H×W×3 RGB) or path to image file
        disease  : one of EyeDiseaseSimulator.SUPPORTED
        severity : float in [0, 1] — 0 = healthy, 1 = most severe

        Returns
        -------
        np.ndarray — transformed image (H×W×3 uint8)
        """
        image = self._load(image)
        if disease not in FILTER_MAP:
            raise ValueError(f"Disease '{disease}' not supported. Options: {self.SUPPORTED}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")

        return FILTER_MAP[disease](image, severity)

    def show_comparison(
        self,
        image: Union[np.ndarray, str, Path],
        disease: str,
        severity: float,
        save_path: Optional[str] = None,
    ) -> None:
        """Show side-by-side: original vs. disease-simulated image."""
        image = self._load(image)
        simulated = self.simulate(image, disease, severity)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f0f0f")
        fig.suptitle(
            f"{disease} — Severity: {severity:.0%}",
            color="white", fontsize=14, fontweight="bold"
        )

        for ax, img, title in zip(
            axes,
            [image, simulated],
            ["Normal Vision", f"{disease} Simulated"],
        ):
            ax.imshow(img)
            ax.set_title(title, color="white", fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def show_progression(
        self,
        image: Union[np.ndarray, str, Path],
        disease: str,
        n_steps: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Show a disease progression strip from healthy to most severe,
        in n_steps evenly spaced severity levels.
        """
        image = self._load(image)
        stages = DISEASE_STAGES.get(disease, [f"Stage {i}" for i in range(n_steps)])
        severities = np.linspace(0, 1, n_steps)

        fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 3.2, 4), facecolor="#0f0f0f")
        fig.suptitle(
            f"{disease} — Progression", color="white", fontsize=14, fontweight="bold"
        )

        for i, (ax, sev) in enumerate(zip(axes, severities)):
            sim = self.simulate(image, disease, sev)
            ax.imshow(sim)
            label = stages[i] if i < len(stages) else f"Severity {sev:.2f}"
            ax.set_title(label, color="white", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    def show_all_diseases(
        self,
        image: Union[np.ndarray, str, Path],
        severity: float = 0.6,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Show all supported disease simulations at a given severity
        in a single grid.
        """
        image = self._load(image)
        diseases = self.SUPPORTED
        n = len(diseases) + 1  # +1 for normal

        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), facecolor="#0f0f0f")
        axes = axes.flatten()

        axes[0].imshow(image)
        axes[0].set_title("Normal Vision", color="white", fontsize=10)
        axes[0].axis("off")

        for i, disease in enumerate(diseases):
            sim = self.simulate(image, disease, severity)
            axes[i + 1].imshow(sim)
            axes[i + 1].set_title(disease, color="white", fontsize=10)
            axes[i + 1].axis("off")

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"All Disease Simulations — Severity: {severity:.0%}",
            color="white", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    # ------------------------------------------------------------------
    # Trajectory visualization (links to progression model output)
    # ------------------------------------------------------------------

    def show_trajectory_frames(
        self,
        image: Union[np.ndarray, str, Path],
        disease: str,
        ages: np.ndarray,
        severity_scores: np.ndarray,
        selected_ages: Optional[list[int]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize vision at selected ages along a disease trajectory.

        Parameters
        ----------
        image           : input image
        disease         : disease name
        ages            : array of ages from trajectory simulation
        severity_scores : expected severity at each age (derived from stage probs)
        selected_ages   : list of ages to visualize (default: 5 evenly spaced)
        """
        image = self._load(image)

        if selected_ages is None:
            idx = np.linspace(0, len(ages) - 1, 5, dtype=int)
            selected_ages = ages[idx].tolist()

        n = len(selected_ages)
        fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 4), facecolor="#0f0f0f")
        if n == 1:
            axes = [axes]

        fig.suptitle(
            f"{disease} — Vision Trajectory",
            color="white", fontsize=14, fontweight="bold"
        )

        for ax, age in zip(axes, selected_ages):
            idx = np.argmin(np.abs(ages - age))
            sev = float(severity_scores[idx])
            sev = np.clip(sev, 0, 1)
            sim = self.simulate(image, disease, sev)
            ax.imshow(sim)
            ax.set_title(f"Age {age}\nSeverity {sev:.2f}", color="white", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.show()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _load(image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load and normalize an image to RGB uint8 np.ndarray."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Image not found: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise TypeError(f"Expected ndarray or path, got {type(image)}")
        return img

    @staticmethod
    def load_sample_image(size: tuple = (500, 750)) -> np.ndarray:
        """
        Generate a synthetic sample scene for testing (no file needed).
        Creates a colorful cityscape-like scene.
        """
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(h // 2):
            t = y / (h // 2)
            img[y, :] = [
                int(30 + t * 100),
                int(80 + t * 120),
                int(180 + t * 60),
            ]

        # Ground
        img[h // 2:, :] = [60, 90, 50]

        # Buildings
        buildings = [(50, 300, 80, 250), (160, 280, 60, 220),
                     (280, 320, 90, 260), (420, 310, 70, 240),
                     (550, 290, 100, 200), (680, 330, 80, 270)]
        for (x, bh, bw, col) in buildings:
            color = [col, col - 20, col - 40]
            cv2.rectangle(img, (x, h // 2 - bh), (x + bw, h // 2), color, -1)
            # Windows
            for wy in range(h // 2 - bh + 15, h // 2 - 20, 25):
                for wx in range(x + 10, x + bw - 10, 18):
                    win_color = [230, 220, 150] if np.random.random() > 0.3 else [40, 40, 60]
                    cv2.rectangle(img, (wx, wy), (wx + 10, wy + 15), win_color, -1)

        # Sun
        cv2.circle(img, (w - 80, 60), 40, (255, 240, 100), -1)

        return img