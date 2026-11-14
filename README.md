# marswalk-model

Walking-model data generator and step-cost surrogate for **Marswalk** (NASA Space Apps, Interplanetary Survival Guide: Martian Map).

A muscle-driven human leg model (MyoLeg, 80 muscles) walks across short Mars terrain **patches**. Two walkers drive it:

| `--walker` | Controller | Image |
|---|---|---|
| `myoleg-deprl` | DEP-RL pretrained myoLegWalk policy (MyoSuite 3.1) | `Dockerfile` (Python 3.11) |
| `kinesis` | KINESIS pretrained directional-control policy (EPFL, ICRA 2026). See [kinesis_backend/README.md](kinesis_backend/README.md) | `Dockerfile.kinesis` (Python 3.8) |

Each run on a patch is turned into labels such as energy cost, falls and stability. A gradient-boosted model then learns *terrain + suit → step cost*, and the Marswalk route planner uses it in place of the hand-made Pandolf + risk formula.

```
generate patches ──► simulate walker ──► labels ──► split by map ──► train surrogate ──► planner lookup
 (terrain.py)         (env.py, runner.py)            (train.py)       (train.py)          (planner_lookup_mars.json)
```

The runs use short patches instead of whole paths. A whole-path run gives one label for hundreds of steps and is wasted if the walker falls halfway. The planner already adds steps together, so it only needs a good cost per step.

## Status

| Part | Tested |
|---|---|
| Terrain patch → MuJoCo heightfield (heights checked with ray casts, every walking axis, MuJoCo 3.17 and 3.1.2), gravity, suit mass, foot contacts | ✅ `tests/test_env.py`, `tests/test_placement.py` |
| KINESIS XML patch (heightfield injected, floor disabled, body lands on terrain) | ✅ on a stand-in model |
| Full generate → Parquet pipeline (with a non-walking `zero` policy) | ✅ |
| Training, grouped split, metrics, planner lookup (on synthetic labels) | ✅ `tests/test_train.py` |
| **Either pretrained policy actually walking** | ❌ not yet. The weights come from Google Drive (DEP-RL) and Hugging Face (KINESIS), which the dev sandbox could not reach. **Run `gcp/submit.sh check` first.** |
| Docker image builds | ❌ not yet. They run on Cloud Build, and each build runs the tests. |

**Known risk:** both policies were trained on flat Earth ground with no suit. The DEP-RL baseline was trained in 2023 on flat ground, at Earth gravity, with no suit. It may not walk in the current MyoSuite at all. If it walks, expect many falls at Mars gravity with a 40 kg pack. `check_walker` tells you which case you are in before you spend compute. See *If the walker does not walk* below.

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

bash gcp/submit.sh build               # APIs, bucket, Artifact Registry, MyoLeg image (~15–25 min)
bash gcp/submit.sh build-kinesis       # KINESIS image (~20–40 min)

# 1 task each: flat Earth -> flat Mars -> Mars + 40 kg suit -> slope sweeps vs Minetti
bash gcp/submit.sh check
WALKER=kinesis bash gcp/submit.sh check
# read gs://$BUCKET/checks/<name>.json ("verdict" line) and pick the walker that walks

WALKER=kinesis TASKS=50 PARALLEL=25 bash gcp/submit.sh generate   # 50 × 4 maps × 8 patches = 1,600 patches
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
python scripts/generate.py --out data/smoke --num-maps 1 --patches-per-map 2 --walker myoleg-zero   # pipeline check
python scripts/check_walker.py --walker myoleg-deprl                                                # downloads weights once
```

## If the walker does not walk

Try these in order:
1. **Pin an older MyoSuite** that matches the 2023 baseline (the observation layout may have changed). Then re-run `check`.
2. **Fine-tune with DEP-RL**: start from the baseline and train on `myoLegRoughTerrainWalk-v0` with gravity and pack mass randomised over the ranges above (`python -m deprl.main <config.yaml>`). A GPU VM helps here.
3. Use the **KINESIS** walker (`--walker kinesis`), or add another backend (SCONE, a MuJoCo Playground humanoid). A backend only needs the small interface described at the top of `marswalk_sim/env.py`.

## Analysis interface (desktop GUI)

A Qt (PySide6) app for looking at how development is going: the generated data, the trained surrogate and the walker check. It is optional and is not part of the Docker images.

```bash
pip install -r requirements.txt -r requirements-gui.txt
python -m marswalk_gui                       # then use the tabs, or pass paths:
python -m marswalk_gui --data data/run-001 --model models/run-001 --check check_walker.json
```

| Tab | What it shows |
|---|---|
| Dataset | Folder of `part-*.parquet` (local or `gs://`): status counts, fall rate, Mars vs Earth check rows, column fill rates, rows |
| Explorer | Scatter, binned mean or histogram of any numeric column, coloured by outcome or a column; filters for status, falls, planet |
| Terrain patch | Build a patch with `marswalk_sim.terrain` from the sampling knobs and see the height map, centre-line profile and the DTM-style stats the model gets |
| Model runs | Add one or more `scripts/train.py` output folders; compares test scores, flags small test sets and train/test gaps; predicted vs simulated on the held-out maps |
| What-if | Move the 11 model features and read cost, fall probability and margin of stability, with slope sweeps against Minetti 2002 |
| Planner lookup | The `planner_lookup_mars.json` grid as heat map and curves |
| Walker check | `check_walker.json`: stage table, verdict, cost by grade vs Minetti |

**Load synthetic demo** (Dataset tab) writes made-up labels from a Pandolf-like formula and trains on them, so you can try the app without a walking policy. A banner marks the demo; it is not simulation output and not Mars measurements.

## Layout

```
marswalk_sim/
  config.py     ranges, constants
  terrain.py    PatchParams, heightfield builder, DTM-style terrain stats
  placement.py  puts a heightfield into any MuJoCo model along ±x / ±y
  env.py        MyoSuite wrapper: terrain, gravity, friction, pack mass; disables the flat floor
  policies.py   DEP-RL | zero | random controllers for the MyoSuite walker
  walkers.py    --walker name → (env, policy)
  runner.py     one rollout → labels (any walker)
kinesis_backend/
  kinesis_env.py  KINESIS env subclass + policy wrapper (Python 3.8)
scripts/
  generate.py   one shard → Parquet (Cloud Batch entry point)
  check_walker.py  flat Earth → flat Mars → Mars + suit → slope sweeps vs Minetti 2002
  train.py      grouped split, gradient boosting, metrics, planner lookup
gcp/            Batch job template + submit.sh
Dockerfile      MyoLeg / DEP-RL image      Dockerfile.kinesis   KINESIS image
marswalk_gui/   analysis interface (core.py has the Qt-free logic, app.py the windows)
tests/          env, training and GUI tests (also run during the Docker build)
```

## Next steps

* Floquet multipliers: generate with `--save-traj` on flat, steady patches, then compute offline.
* Real terrain: cut 16 × 12 m windows from a HiRISE DTM of the landing site into `Heightfield` objects (the `dtm_*` features stay the same).
* Calibrate the energy proxy with analog EVA data (the `k_suit` factor in the planner).

## References

* MyoSuite (Caggiano et al., 2022) and the Rajagopal 2016 leg model
* DEP-RL (Schumacher et al., 2023) pretrained myoLegWalk baseline
* KINESIS (Simos, Chiappa, Mathis, ICRA 2026), github.com/amathislab/kinesis
* Minetti et al. 2002, energy cost of walking on gradients (`check_walker.py`)
* Hof et al. 2005, margin of stability (extrapolated centre of mass)
* Pandolf 1977 / Santee 2003: the analytic model this surrogate replaces
