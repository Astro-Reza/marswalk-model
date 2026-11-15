#!/usr/bin/env bash
# Set up KINESIS + marswalk-model on a fresh Ubuntu machine (EC2, or WSL) for finetune_mars.py.
# Same steps as Dockerfile.kinesis, without Docker and without root for Python (uv in ~/.local).
#
#   bash setup_ec2.sh [marswalk-branch]
#
# Then copy your checkpoint to ~/kinesis/data/trained_models/legs/target-reach-mars/model.pth
# and run the command printed at the end (inside tmux, so an SSH drop does not stop it).
set -euo pipefail
BRANCH=${1:-main}
KINESIS_COMMIT=bfee51db8207c6f494a278acc04094b59f3d4fad
KDIR="$HOME/kinesis"
MDIR="$HOME/marswalk-model"

if sudo -n true 2>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git curl tmux build-essential libgl1 libglib2.0-0 >/dev/null
fi

command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d "$KDIR/.git" ] || git clone -q https://github.com/amathislab/kinesis "$KDIR"
git -C "$KDIR" checkout -q "$KINESIS_COMMIT"
[ -d "$MDIR/.git" ] || git clone -q https://github.com/Astro-Reza/marswalk-model "$MDIR"
git -C "$MDIR" fetch -q origin "$BRANCH" && git -C "$MDIR" checkout -q "$BRANCH"
[ -f "$MDIR/kinesis_backend/finetune_mars.py" ] || { echo "finetune_mars.py not on branch $BRANCH: push it first"; exit 1; }

cd "$KDIR"
[ -x .venv/bin/python ] || uv venv -q --python 3.8 .venv
PY=.venv/bin/python
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then TORCH=cu117; else TORCH=cpu; fi
uv pip install -q --python $PY pip setuptools wheel
uv pip install -q --python $PY "torch==1.13.1+$TORCH" "torchvision==0.14.1+$TORCH" \
  --index-url "https://download.pytorch.org/whl/$TORCH" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install -q --python $PY numpy==1.21.6
uv pip install -q --python $PY --no-build-isolation chumpy==0.70
uv pip install -q --python $PY -r requirements.txt
uv pip install -q --python $PY huggingface_hub pyarrow==12.0.1 pytest

[ -d data/xml ] || $PY src/utils/download_assets.py --branch kinesis-2.0
[ -f data/trained_models/legs/target-reach/model.pth ] || $PY src/utils/download_models.py
mkdir -p data/trained_models/legs/target-reach-mars

$PY -c "import torch, mujoco; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
echo "vCPUs: $(nproc)   RAM: $(free -g | awk 'NR==2{print $2}') GB"
cat <<EOF

Ready. Copy model.pth into $KDIR/data/trained_models/legs/target-reach-mars/ (else training starts from the pretrained policy), then:
  tmux new -s ft
  cd $MDIR && KINESIS_DIR=$KDIR $PY kinesis_backend/finetune_mars.py --epochs 300 --threads $(( $(nproc) - 2 )) -- learning.min_batch_size=51200 learning.save_curr_frequency=10 2>&1 | tee finetune.log
EOF
