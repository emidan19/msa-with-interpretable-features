# MSA Feature Extraction Prototype

Prototype for extracting interpretable and SSL-based features for music
structure analysis, then converting them into beat-synchronous descriptors and
self-similarity matrices.

## Copy repo

```bash
git clone https://github.com/emidan19/msa-with-interpretable-features.git
cd msa-with-interpretable-features/
```

## Environment

The environment used in this workspace is:

```bash
conda activate msa_features
```

To recreate it:

```bash
conda env create -f environment.yml
conda activate msa_features
```

## Modules

- `data_loading.py`: loads YAML config, audio files, section annotations, and demo audio.
- `beat_tracking.py`: estimates beats/downbeats and regularizes beat boundaries.
- `feature_types.py`: computes descriptor streams and SSM-ready feature matrices, including SSL embeddings.
  - _IMPORTANT NOTE:_ currently this code takes all the SSL models within the same folder as a "grandparent" directory. For example, if running the script: 
  ```python 
  python extract_msa_features.py
  ``` 
  make sure to have the SSL models repos ([MusicFM](https://github.com/minzwon/musicfm), [MuQ](https://github.com/tencent-ailab/MuQ/), [MATPAC](https://github.com/aurianworld/matpac/), or any other you want) at ``msa-with-interpretable-features/../../SSL_models``
  - The `repcl` option instead loads the repetition-based triplet-mining model from [Triplet Mining](https://github.com/morgan76/Triplet_Mining/tree/main)
  - The `mule` option loads the TensorFlow/Keras MULE model from [music-audio-representations](https://github.com/PandoraMedia/music-audio-representations) (`supporting_data/model/model.keras`, fetched with `git lfs pull`; requires `sxmp-mule`). Configure it with `mule_model_path`, `mule_hop_frames` (embedding hop in 10 ms mel frames, default 50 = 0.5 s) and `mule_batch_size`.
- `pipeline.py`: orchestrates extraction, beat-synchronous aggregation, preview rendering, and output writing.

## Config File

The extraction scripts use `msa_feature_config.template.yaml`.

Important fields:

- `audio_path`: single-file mode.
- `audio_dir`: batch mode.
- `out_dir`: output directory for `.npz`, preview images, and summaries.
- `feature_config.computing_features`: only these features are computed.
- `feature_config.preview_features`: only these computed features are previewed.
- `feature_config.beat_tracking_method`: `librosa`, `beat_this`, or `madmom`.

Example feature section:

```yaml
feature_config:
  computing_features:
    - stm
    - mfcc
    - chroma
    - matpac
    - muq
    - musicfm
    - repcl
  preview_features:
    - stm
    - mfcc
    - chroma
    - matpac
```

## Usage

Single-file extraction with config:

```bash
python extract_msa_features.py --config msa_feature_config.template.yaml
```

Single-file extraction with an explicit audio path:

```bash
python extract_msa_features.py path/to/song.wav --config msa_feature_config.template.yaml
```

Batch extraction with config:

```bash
python batch_extract_msa_features.py --config msa_feature_config.template.yaml
```

The internal orchestration happens through `pipeline.py`, while
`extract_msa_features.py` and `batch_extract_msa_features.py` are the public
CLI entrypoints.

## MSA Principles Measures (LAMIR 2026)

For the study of the MSA measures of homogeneity, repetition, and novelty,
see the README at
[experiments/msa_principles_measure/README.md](experiments/msa_principles_measure/README.md).

## Outputs

Each run can produce:

- raw feature arrays for the requested features
- beat-synchronous feature arrays
- SSMs for the requested computed features
- preview images for the requested preview features
- a JSON summary with shapes, diagnostics, and output paths

## Downstream Structure Baseline

`run_structure_baseline.py` segments each track from its SSMs with the
configured algorithms and writes one JAMS file per track, algorithm and SSM run.
It also embeds the ground-truth segmentation (when available) in each JAMS, so
`evaluate_structure_baseline.py` can score the outputs without the original
section files.

```bash
python run_structure_baseline.py --config path/to/config.yaml \
  --feature-dir feature_outputs/ \
  --experiment-dir experiments/structure_baseline \
  --limit 20 \
  --resume
```

CLI arguments (each overrides the matching YAML value):

- `--config`: YAML config path. Segmentation settings live under the
  `segmentation` key: `algorithms`, `ssm_features`, `graphditty_fusion.features`,
  `feature_dir` and `experiment_dir`.
- `--feature-dir`: folder with precomputed `*_features.npz` files. If none are
  found, features are extracted from `audio_dir` (set in the YAML), and saved to
  `out_dir` only if it is set.
- `--audio-dir`: folder of audio files for on-the-fly feature extraction.
- `--experiment-dir`: output folder for JAMS files and summaries
  (default `experiments/structure_baseline`).
- `--limit`: process only the first N tracks.
- `--sr`, `--hop-length`, `--beat-tracking-method` (`librosa`, `beat_this`,
  `madmom`): feature extraction overrides, used only when extracting from audio.
- `--resume`: skip tracks whose JAMS outputs already exist for every
  algorithm / SSM run, keeping their rows from the previous summary.

Outputs in the experiment dir:

- `<algorithm>__<ssm_run>/<track_id>.jams`: estimated segmentation, per-segment
  CBM metrics, and the reference annotation.
- `fusion_ssms/`: fused SSMs for GraphDitty multi-feature runs.
- `segmentation_summary.json` and `.csv`: one row per track, algorithm and SSM run.
tesis/msa/MSA-with-Interpretable-Features