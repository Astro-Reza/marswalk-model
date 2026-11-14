# KINESIS walker backend

[KINESIS](https://github.com/amathislab/kinesis) (Simos, Chiappa, Mathis, EPFL, ICRA 2026) drives the same 80-muscle MyoLeg body as the MyoSuite walker. Its muscle activity correlates with human EMG. Code is BSD-3; the pretrained models on Hugging Face are MIT.

## How it is used here

KINESIS is a motion-imitation policy. Its **directional control** mode gives the imitation policy a goal ~1.3 m ahead of the pelvis on every step, so it walks without any reference clip. We use that mode, which means:

* **no AMASS / KIT motion data and no SMPL model** are needed (the motion library is replaced by a stub)
* the pretrained `target-reach` checkpoint (shared by target reaching and directional control) is loaded from `data/trained_models/<model>/target-reach/model.pth`

`kinesis_env.py` subclasses the KINESIS environment without editing the KINESIS repo:

| Change | Why |
|---|---|
| injects a 100 × 100 heightfield geom into the model XML, copies the floor's contact settings to it, disables the flat floor | terrain patches, and no invisible floor under downhill ground |
| root height (observation and the < 0.7 m fall check) is measured from the terrain under the pelvis | otherwise walking downhill looks like falling, and uphill looks like floating |
| gravity, friction and suit/PLSS mass set per patch | Mars conditions |
| walking direction fixed (goal toward world −x); 3 m flat run-in | the walker may turn to face the goal first |

Labels are computed by the same `marswalk_sim/runner.py` as the MyoLeg walker, so both walkers write the same Parquet columns (`walker` column tells them apart).

## Limits to keep in mind

* The policy learned to imitate **Earth, flat-ground** motion capture. On Mars slopes it will try to keep an Earth-like gait. How far it can stretch, and where it falls, is exactly what `check_walker.py` measures.
* The `root_height` change is something the policy never saw in training. On flat ground it is identical to the original, so stage 1 of the check is a fair test. On slopes it is the most sensible input, but it is still new to the policy.
* Not run end to end yet: the KINESIS assets and weights come from Hugging Face, which the dev sandbox could not reach. The placement and XML-patch code is tested (`tests/test_placement.py`, also on MuJoCo 3.1.2).

## Run

On Google Cloud:

```bash
bash gcp/submit.sh build-kinesis          # ~20–40 min the first time
WALKER=kinesis bash gcp/submit.sh check   # flat Earth -> flat Mars -> Mars + 40 kg suit -> slope sweeps
WALKER=kinesis TASKS=50 bash gcp/submit.sh generate
```

Locally (Linux, conda):

```bash
git clone https://github.com/amathislab/kinesis /opt/kinesis && cd /opt/kinesis
git checkout bfee51db8207c6f494a278acc04094b59f3d4fad
conda create -n kinesis python=3.8 && conda activate kinesis
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.21.6 && pip install --no-build-isolation chumpy==0.70 && pip install -r requirements.txt
pip install huggingface_hub pyarrow==12.0.1
python src/utils/download_assets.py --branch kinesis-2.0 && python src/utils/download_models.py
cd /path/to/marswalk-model
KINESIS_DIR=/opt/kinesis python scripts/check_walker.py --walker kinesis --repeats 2
```

Environment variables: `KINESIS_DIR` (checkout, default `/opt/kinesis`), `KINESIS_MODEL` (`legs` default, `legs_abs`, `legs_back`), `KINESIS_CHECKPOINT_DIR` (override the checkpoint folder).
