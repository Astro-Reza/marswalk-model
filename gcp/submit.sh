#!/usr/bin/env bash
# Build the image and submit a Google Cloud Batch job.
#
#   export PROJECT=my-gcp-project BUCKET=marswalk-data-123
#   bash gcp/submit.sh build          # one time: enable APIs, create bucket + registry, build image
#   bash gcp/submit.sh check          # 1 task: does the walker walk? Earth vs Minetti, Mars sweep
#   bash gcp/submit.sh generate       # N tasks: simulation shards -> gs://$BUCKET/runs/$RUN_ID
#
# Optional: REGION (default asia-southeast1), RUN_ID, TASKS (default 50), PARALLEL (25),
#           MAPS_PER_TASK (4), PATCHES_PER_MAP (8), MACHINE (e2-standard-2), SPOT (true)
set -euo pipefail

: "${PROJECT:?set PROJECT}"
: "${BUCKET:?set BUCKET (globally unique bucket name, no gs://)}"
REGION="${REGION:-asia-southeast1}"
REPO="${REPO:-marswalk}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/sim:latest"
RUN_ID="${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)}"
TASKS="${TASKS:-50}"
PARALLEL="${PARALLEL:-25}"
MAPS_PER_TASK="${MAPS_PER_TASK:-4}"
PATCHES_PER_MAP="${PATCHES_PER_MAP:-8}"
MACHINE="${MACHINE:-e2-standard-2}"
PROVISIONING=$([[ "${SPOT:-true}" == "true" ]] && echo SPOT || echo STANDARD)
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

gcloud config set project "$PROJECT" >/dev/null

make_job() {  # $1 = job name, $2 = JSON array of commands, $3 = tasks, $4 = parallelism
  local out="$HERE/job-$1.json"
  sed -e "s|__IMAGE__|${IMAGE}|" -e "s|__BUCKET__|${BUCKET}|" -e "s|__COMMANDS__|$2|" \
      -e "s|__TASKS__|$3|" -e "s|__PARALLEL__|$4|" -e "s|__MACHINE__|${MACHINE}|" \
      -e "s|__PROVISIONING__|${PROVISIONING}|" "$HERE/job_template.json" > "$out"
  echo "$out"
}

case "${1:-}" in
  build)
    gcloud services enable batch.googleapis.com artifactregistry.googleapis.com \
      cloudbuild.googleapis.com compute.googleapis.com logging.googleapis.com
    gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
      gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"
    gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 || \
      gcloud storage buckets create "gs://$BUCKET" --location "$REGION"
    gcloud builds submit "$ROOT" --tag "$IMAGE" --machine-type=e2-highcpu-8 --timeout=3600s
    ;;
  check)
    name="check-$(date +%Y%m%d-%H%M%S)"
    cmds="[\"scripts/check_walker.py\", \"--policy\", \"deprl\", \"--repeats\", \"3\", \"--out\", \"/mnt/disks/out/checks/${name}.json\"]"
    job=$(make_job "$name" "$cmds" 1 1)
    gcloud batch jobs submit "$name" --location "$REGION" --config "$job"
    echo "Result: gs://$BUCKET/checks/${name}.json  (logs: Cloud Logging, job $name)"
    ;;
  generate)
    cmds="[\"scripts/generate.py\", \"--out\", \"/mnt/disks/out/runs/${RUN_ID}\", \"--num-maps\", \"${MAPS_PER_TASK}\", \"--patches-per-map\", \"${PATCHES_PER_MAP}\", \"--policy\", \"deprl\"]"
    job=$(make_job "$RUN_ID" "$cmds" "$TASKS" "$PARALLEL")
    gcloud batch jobs submit "$RUN_ID" --location "$REGION" --config "$job"
    echo "Data: gs://$BUCKET/runs/${RUN_ID}  ($((TASKS * MAPS_PER_TASK * PATCHES_PER_MAP)) patches)"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
