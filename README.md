# VisioGen — Genetic Eye Disease Modeling & Vision Simulation

> *A personalized, uncertainty-aware pipeline from genetic risk to visual experience.*

---

## Overview

**VisioGen** is a three-part computational project that models eye disease genetically, simulates disease progression mathematically, and renders a personalized vision trajectory — showing what the world looks like through the eyes of someone with a given genetic risk profile, at any point in their life.

Unlike existing visual simulators that apply static image filters, VisioGen couples a **Polygenic Risk Score (PRS) model** with a **Bayesian Markov progression model** to generate dynamic, uncertainty-aware visual outputs that evolve with disease severity over time.

---

## Motivation

Most tools in this space do one thing: show a blurry photo labeled "this is what cataracts look like." VisioGen asks a harder question:

> *Given who you are genetically and how you live, what will you see at age 50? At 65? At 80 — and with what uncertainty?*

The project sits at the intersection of statistical genetics, mathematical modeling, and visual communication.

---

## Project Structure

```
visiogen/
│
├── 01_genetics/              # Polygenic Risk Score modeling
│   ├── prs_pipeline.py       # PRS construction from GWAS summary stats
│   ├── gxe_model.py          # Gene-environment interaction modeling
│   └── ancestry_portability/ # PRS accuracy across ancestries
│
├── 02_progression/           # Disease progression modeling
│   ├── bayesian_markov.py    # Bayesian Markov chain with uncertainty
│   ├── coupled_model.py      # Co-occurring disease progression
│   └── survival_model.py     # Cox proportional hazards baseline
│
├── 03_simulation/            # Visual simulation pipeline
│   ├── disease_filters/      # Per-disease image transformations
│   ├── uncertainty_render.py # Ensemble/probabilistic visualization
│   ├── gaze_simulation.py    # Foveal fixation & gaze-contingent rendering
│   └── trajectory.py         # Full vision timeline generator
│
├── data/                     # Data sources and preprocessing scripts
│   └── README_data.md        # Instructions for accessing GWAS Catalog, UK Biobank
│
├── notebooks/                # Exploratory analysis and walkthroughs
└── app/                      # (Planned) Interactive Streamlit interface
```

---

## The Three-Part Pipeline

### Part 1 — Genetic Risk Modeling

Polygenic Risk Scores (PRS) aggregate the effect of thousands of genetic variants from GWAS data into a single, interpretable risk estimate per individual.

**Key features:**
- PRS construction using GWAS summary statistics (GWAS Catalog)
- Gene-environment interaction (GxE) terms: genetic risk modulated by lifestyle covariates (UV exposure, smoking, diabetes status, BMI)
- Ancestry portability analysis: quantifying PRS accuracy degradation as a function of genetic distance from the training population
- Output: a continuous individual risk score *r ∈ [0, 1]* per disease

**Anchor disease:** Age-related Macular Degeneration (AMD) — richest GWAS data, clean staging, strong genetic signal (*CFH*, *ARMS2*)

---

### Part 2 — Bayesian Markov Progression Model

Disease progression is modeled as a discrete Markov chain over severity stages, with transition probabilities treated as probability distributions rather than fixed scalars.

**Stages (AMD example):**
```
Healthy → Early AMD → Intermediate AMD → Advanced AMD → Legal Blindness
```

**Key features:**
- Transition probabilities parameterized as Beta distributions, with priors from clinical literature
- Genetic risk score *r* and GxE covariates modulate transition rates
- Full uncertainty propagation: posterior distributions over disease stage at every age
- Coupled disease modeling: correlated Markov chains for co-occurring conditions (e.g. AMD + diabetic retinopathy)
- Output: a **probability distribution over disease stages** as a function of age

---

### Part 3 — Visual Simulation

Each disease stage is mapped to a parameterized image transformation. The simulation renders not one image, but a **distribution of visual outcomes** consistent with the model's uncertainty.

**Supported diseases:**

| Disease | Visual Signature | Transformation |
|---|---|---|
| AMD | Central scotoma | Foveal Gaussian mask, contrast reduction |
| Glaucoma | Peripheral field loss | Radial vignette, tunnel masking |
| Diabetic Retinopathy | Floaters, hemorrhages | Stochastic dark spot overlay, blur |
| Cataracts | Diffuse haze, halos | Global blur, contrast/gamma reduction |
| Retinitis Pigmentosa | Ring scotoma → tunnel | Annular mask progressing inward |
| Color Vision Deficiency | Color remapping | Linear deuteranopia/protanopia transform |

**Novel features:**
- **Uncertainty-aware rendering:** visual outputs are drawn from the posterior distribution over disease stages, visualized as an animated ensemble or confidence-interval overlay
- **Gaze-contingent simulation:** scotomas and degradations follow a simulated fixation point, reflecting real perceptual experience more accurately than static masks
- **Vision trajectory:** a continuous video or interactive timeline of visual degradation from age 20 to 80, given a genetic risk profile

---

## Novelty

Most existing tools apply one static filter to one image for one disease at one severity level. VisioGen is differentiated by:

1. **End-to-end coupling** — genetics → progression → vision, in one coherent pipeline
2. **Uncertainty quantification** — Bayesian treatment of progression propagated into visual output
3. **GxE modeling** — environmental factors modulate genetic risk, not just add to it
4. **Dynamic trajectory** — vision changes over a lifetime, not just at one snapshot
5. **Gaze-contingent rendering** — physiologically realistic, not cosmetic

---

## Data Sources

| Source | Used For |
|---|---|
| [GWAS Catalog](https://www.ebi.ac.uk/gwas/) | Variant effect sizes for PRS construction |
| [UK Biobank](https://www.ukbiobank.ac.uk/) | Individual-level genotype + eye phenotype data |
| [REFUGE / ORIGA](https://refuge.grand-challenge.org/) | Retinal imaging with disease labels |
| Clinical literature (meta-analyses) | Markov transition probability priors |

> See `data/README_data.md` for access instructions and preprocessing steps.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Genetic modeling | Python, NumPy, pandas, scikit-learn, statsmodels |
| Bayesian modeling | PyMC, SciPy |
| Image simulation | OpenCV, Pillow, Matplotlib |
| Visualization | Plotly, Matplotlib, HTML5 Canvas (gaze simulation) |
| App (planned) | Streamlit or Gradio |

---

## Roadmap

- [x] Project architecture and documentation
- [ ] PRS pipeline for AMD (GWAS Catalog → risk score)
- [ ] GxE interaction model
- [ ] Bayesian Markov chain for AMD progression
- [ ] Visual simulation layer (AMD + Glaucoma)
- [ ] Uncertainty-aware rendering
- [ ] Extend simulation to remaining diseases
- [ ] Gaze-contingent simulation (JavaScript/canvas)
- [ ] Coupled disease model (AMD + diabetic retinopathy)
- [ ] Vision trajectory video generator
- [ ] Interactive web app

---

## About

This is an independent undergraduate research project developed at the intersection of mathematical modeling, statistical genetics, and visual communication. It is not affiliated with any institution.

Feedback, collaboration, and suggestions are welcome — open an issue or reach out directly.

---

*Built with curiosity. Modeled with rigor. Rendered with care.*
