# MSA Principles Measure

Experiments that measure how well each feature's self-similarity matrix (SSM)
follows the three MSA principles (homogeneity, repetition and novelty) with
respect to the ground-truth section annotations. They also check whether those
scores are significant compared with randomly simulated section layouts.

The pipeline has four steps:

1. **Feature extraction** (`batch_extract_msa_features.py`): builds the
   `*_features.npz` bundles with beat-synchronous features and the
   ground-truth section SSM.
2. **Similarity function comparison** (`by_similarity_function.py`): scores
   each feature's SSM under different similarity functions.
3. **Distribution validation** (`distribution_validation.py`): compares the
   scores against the ground-truth sections with the scores against simulated
   sections.
4. **Divergence measures** (`compute_div_measures.py`): computes the KL
   divergence between the GT-score and simulated-score distributions, plus
   summary statistics.

## Setup

```bash
conda activate msa_features   # see the main README to create it
```

All commands below run from the **repository root**.

## 1. Extract features

```bash
python batch_extract_msa_features.py \
  --config configs/dataset_config.template.yaml \
  --out-dir feature_outputs/dataset_features
```

Before you run it, check these fields in the config:

- `audio_dir`: folder that contains the selected dataset `.wav` files.
- `limit`: number of songs to process (`100` for the full RWC-pop dataset).
- `feature_config.sections_dir`: folder with the section annotations.
- `feature_config.computing_features`: must include `sections`, because the
  later steps need the `ssm_sections` entry in each bundle. The reported
  results use `stm`, `cens`, `chroma`, `mfcc`, `muq`, `musicfm` and `repcl`.

The SSL features (`muq`, `musicfm`, `repcl`, ...) need their model repositories
in the location described in the main [README](../../README.md#modules).

Each song produces `<song>_features.npz` plus a preview image. The script also
writes a `batch_summary.csv`/`.json` index in `--out-dir`.

## 2. Compute homogeneity, repetition and novelty for each similarity function

```bash
python experiments/msa_principles_measure/by_similarity_function.py \
  --features-dir feature_outputs/dataset_features \
  --output-dir experiments/msa_principles_measure/similarity_comparison_outputs \
  --similarity-types cosine rbf
```

| Argument | Default | Description |
| --- | --- | --- |
| `--features-dir` | searches `feature_outputs/`, `outputs/`, `out/` | Folder with `*_features.npz` bundles |
| `--output-dir` | `experiments/msa_principles_measure/similarity_comparison_outputs` | Where the CSV and figures are written |
| `--similarity-types` | `cosine rbf` | Similarity functions to compare |
| `--gamma` | `None` | Override for the RBF `gamma` |

Outputs:

- `similarity_comparison_results.csv`: one row per song, feature and similarity
  function, with the homogeneity, repetition and novelty scores, plus the
  per-region mean similarities (`section_mean`, `repetition_mean`,
  `background_mean`) and `global_quality`.
- `<metric>_global_comparison.png`: mean ± std per feature and similarity function.
- `metrics_violin.png`: score distributions per similarity function.
- `<region>_heatmap.png`: mean similarity per region.

## 3. Validate against simulated SSMs

The script estimates the mean/std of the section lengths and the
repetition-to-section ratio from the annotations. It then simulates random
section SSMs with those statistics. For each song it scores the feature SSM
against the ground-truth sections and against `--n-sim` simulated layouts. It
also computes an empirical p-value (the fraction of simulations scoring
greater than or equal to the GT score).

Per-feature validation (used for the reported results):

```bash
python experiments/msa_principles_measure/distribution_validation.py \
  --features-dir feature_outputs/dataset_features \
  --similarity-type cosine \
  --n-sim 100

python experiments/msa_principles_measure/distribution_validation.py \
  --features-dir feature_outputs/dataset_features \
  --similarity-type rbf \
  --n-sim 100
```

| Argument | Default | Description |
| --- | --- | --- |
| `--features-dir` | searches `feature_outputs/`, `outputs/`, `out/` | Folder with `*_features.npz` bundles |
| `--similarity-type` | `None` | `cosine`, `autocorrelation` or `rbf`. If omitted, the script runs the sanity check below |
| `--n-sim` | `4` | Simulated SSMs per song |
| `--random-seed` | `123` | Seed for the simulation and for picking example songs |
| `--num-gt-examples` | `3` | Songs used in the sanity check |
| `--view-sims` | off | Save a figure with GT and simulated SSMs (sanity check only) |
| `--output-file` | `experiments/msa_principles_measure/ssm_simulation_examples.png` | Base output path; per-feature results go to `<its folder>/validation_<similarity>/` |

With `--similarity-type`, the script writes these files to
`validation_<similarity>/` for each feature:

- `ssm_simulation_<feature>_gt_scores.csv`: scores against the GT sections (one row per song).
- `ssm_simulation_<feature>_sim_scores.csv`: scores against the simulated sections (`n_sim` rows per song).
- `ssm_simulation_<feature>_pvalues.csv`: p-value per song.
- `ssm_simulation_<feature>_score_histograms.png`: GT and simulated score histograms.

**Sanity check** (without `--similarity-type`): uses the GT SSM itself as the
"feature" SSM for a few songs. It writes
`ssm_simulation_examples_score_histograms.png` and
`ssm_simulation_examples_pvalues.csv`. With `--view-sims` it also writes
`ssm_simulation_examples.png`:

```bash
python experiments/msa_principles_measure/distribution_validation.py \
  --features-dir feature_outputs/dataset_features \
  --view-sims
```

## 4. Compute divergence measures

`compute_div_measures.py` reads every `ssm_simulation_*_gt_scores.csv` /
`*_sim_scores.csv` pair in a folder. It then estimates `KL(GT || Sim)` for each
measure with a Gaussian KDE. Pass the output folder of step 3 with
`--results-dir`:

```bash
python experiments/msa_principles_measure/compute_div_measures.py \
  --results-dir experiments/msa_principles_measure/validation_cosine

python experiments/msa_principles_measure/compute_div_measures.py \
  --results-dir experiments/msa_principles_measure/validation_rbf
```

| Argument | Default | Description |
| --- | --- | --- |
| `--results-dir` | `experiments/msa_principles_measure/validation` | Folder with the `*_gt_scores.csv` / `*_sim_scores.csv` files from step 3 |

This step needs NumPy >= 2.0 because it uses `np.trapezoid`.

The script writes these files to the same folder:

- `kl_divergences.csv`: KL divergence per feature for homogeneity, repetition and novelty.
- `gt_statistics.csv`: mean, std and median of the GT scores per feature.