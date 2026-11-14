# marswalk-model

Walking-model data generator and step-cost surrogate for **Marswalk** (NASA Space Apps, Interplanetary Survival Guide: Martian Map).

A muscle-driven human leg model (MyoSuite **myoLeg**, 80 muscles) walks across short Mars terrain **patches**. Each run is turned into labels such as energy cost, falls and stability. A gradient-boosted model then learns *terrain + suit → step cost*, and the Marswalk route planner uses it in place of the hand-made Pandolf + risk formula.

```
generate patches ──► simulate walker ──► labels ──► split by map ──► train surrogate ──► planner lookup
 (terrain.py)         (env.py, runner.py)            (train.py)       (train.py)          (planner_lookup_mars.json)
```

The runs use short patches instead of whole paths. A whole-path run gives one label for hundreds of steps and is wasted if the walker falls halfway. The planner already adds steps together, so it only needs a good cost per step.

## Status

| Part | Tested |
|---|---|
| Terrain patch → MuJoCo heightfield (heights checked with ray casts), gravity, suit mass, foot contacts | ✅ `tests/test_env.py` |
| Full generate → Parquet pipeline (with a non-walking `zero` policy) | ✅ |
| Training, grouped split, metrics, planner lookup (on synthetic labels) | ✅ `tests/test_train.py` |
| **DEP-RL pretrained walking policy actually walking in MyoSuite 3.1** | ❌ not yet. Its weights download from Google Drive, which the dev sandbox could not reach. **Run `gcp/submit.sh check` first.** |
| Docker image build | ❌ not yet. It runs on Cloud Build, and the build itself runs the tests. |

**Known risk:** the DEP-RL baseline was trained in 2023 on flat ground, at Earth gravity, with no suit. It may not walk in the current MyoSuite at all. If it walks, expect many falls at Mars gravity with a 40 kg pack. `check_walker` tells you which case you are in before you spend compute. See *If the walker does not walk* below.

## What gets varied per patch (`marswalk_sim/config.py`)

slope −25…+25°, side slope ±10°, roughness amplitude 0–6 cm and wavelength 0.3–2 m, rocks (density, height), friction 0.4–1.0, gravity (Mars 3.711, plus 10 % Earth 9.81 runs as a sanity check), suit + PLSS mass 0–60 kg, pack position (up/down, behind the torso).

Patches are grouped in **maps** (8 per map). Patches in one map share surface (friction, roughness, rocks) and differ in slope. The train/validation/test split is done **by map** so near-duplicate patches never leak across splits.

## Labels (one Parquet row per patch)

| Column | Meaning |
|---|---|
| `cot_net_J_per_kg_m`, `cot_gross_J_per_kg_m` | cost of transport over the patch. Metabolic proxy = positive muscle work / 0.25 + negative work / 1.2 (+ 1.2 W/kg basal for gross) |
| `met_power_net_W` | mean metabolic power proxy |
| `fell` | centre of mass dropped below 0.8 m or the body tipped (MyoSuite termination) |
| `mos_ml_min_m`, `mos_ml_mean_m`, `mos_ap_mean_m` | margin of stability (Hof XCoM vs stance foot), side-to-side and fore-aft |
| `stride_time_mean_s`, `stride_time_cv` | stride timing and variability |
| `peak_grf_bw` | peak ground reaction force / body weight |
| `distance_m`, `speed_mps`, `status` | `ok`, `fell_before_patch`, `too_short`, `error: …` |
| `dtm_*` | slope and roughness **measured from the heights**, like you would from a HiRISE DTM. These are the model's terrain features. |

Energy is a muscle-work proxy, not a measured VO₂. Treat it as relative and calibrate it against analog EVA data.

## Run on Google Cloud

Needs the `gcloud` CLI (Cloud Shell has it) and a project with billing.

```bash
git clone https://github.com/Astro-Reza/marswalk-model && cd marswalk-model
export PROJECT=your-project-id
export BUCKET=marswalk-data-<something-unique>
export REGION=asia-southeast1          # optional

bash gcp/submit.sh build               # APIs, bucket, Artifact Registry, Cloud Build image (~15–25 min)
bash gcp/submit.sh check               # 1 task. Read gs://$BUCKET/checks/<name>.json when it finishes
# only if the check says the policy walks:
TASKS=50 PARALLEL=25 bash gcp/submit.sh generate   # 50 × 4 maps × 8 patches = 1,600 patches
```

* Each task is one shard with its own seed (`BATCH_TASK_INDEX`), so re-runs and retries never duplicate data.
* Output goes to `gs://$BUCKET/runs/<RUN_ID>/part-*.parquet` plus one `meta-*.json` per shard (versions, calibration).
* Spot VMs are on by default (`SPOT=false` to turn off). Retries are set to 2.
* One control step costs ~1 ms of CPU, so a 10 s patch is a few seconds. Scale with `TASKS`, not bigger machines.
* Watch progress in **Batch → Jobs** or Cloud Logging.

### Train

On a small VM, a Vertex AI Workbench notebook, or your laptop:

```bash
python scripts/train.py --data gs://$BUCKET/runs/<RUN_ID> --out models/<RUN_ID>
gcloud storage cp -r models/<RUN_ID> gs://$BUCKET/models/
```

Outputs: `model_cot.joblib`, `model_fall.joblib`, `model_mos.joblib`, `metrics.json` (train/val/test scores), `split.json` (which maps went where), and `planner_lookup_mars.json`. The lookup is a slope × roughness grid of predicted cost and fall probability that the web Step Scorer can interpolate.

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests
python scripts/generate.py --out data/smoke --num-maps 1 --patches-per-map 2 --policy zero   # pipeline check
python scripts/check_walker.py --policy deprl                                                # downloads weights once
```

## If the walker does not walk

Try these in order:
1. **Pin an older MyoSuite** that matches the 2023 baseline (the observation layout may have changed). Then re-run `check`.
2. **Fine-tune with DEP-RL**: start from the baseline and train on `myoLegRoughTerrainWalk-v0` with gravity and pack mass randomised over the ranges above (`python -m deprl.main <config.yaml>`). A GPU VM helps here.
3. Switch the walker to **SCONE** (`sconegym`), which has pretrained gait models. Only `env.py` and `policies.py` change, and the rest of the pipeline stays the same.

## Layout

```
marswalk_sim/
  config.py     ranges, constants
  terrain.py    PatchParams, heightfield builder, DTM-style terrain stats
  env.py        MyoSuite wrapper: writes terrain, gravity, friction, pack mass; disables the flat floor
  policies.py   deprl | zero | random controllers
  runner.py     one rollout → labels
scripts/
  generate.py   one shard → Parquet (Cloud Batch entry point)
  check_walker.py  walks? + Earth slope sweep vs Minetti 2002 + Mars sweep
  train.py      grouped split, gradient boosting, metrics, planner lookup
gcp/            Batch job template + submit.sh
tests/          env and training tests (also run during the Docker build)
```

## Next steps

* Floquet multipliers: generate with `--save-traj` on flat, steady patches, then compute offline.
* Real terrain: cut 16 × 12 m windows from a HiRISE DTM of the landing site into `Heightfield` objects (the `dtm_*` features stay the same).
* Calibrate the energy proxy with analog EVA data (the `k_suit` factor in the planner).

## References

* MyoSuite (Caggiano et al., 2022) and the Rajagopal 2016 leg model
* DEP-RL (Schumacher et al., 2023) pretrained myoLegWalk baseline
* Minetti et al. 2002, energy cost of walking on gradients (`check_walker.py`)
* Hof et al. 2005, margin of stability (extrapolated centre of mass)
* Pandolf 1977 / Santee 2003: the analytic model this surrogate replaces
