#!/usr/bin/env bash
# Entrena PredRNN v2.
#
# Uso:
#   scripts/train.sh <base|dmref> [args extra para train.py]
#
# Ejemplos:
#   scripts/train.sh base
#   scripts/train.sh dmref --max_iterations 20000 --lr 0.0005
#   BATCH_SIZE=4 PATCH_SIZE=8 IMG_HEIGHT=408 scripts/train.sh dmref
#   PRETRAINED=checkpoints/dmref_dmref_predrnn_v2 scripts/train.sh dmref   # reanudar
#
# Variables de entorno (opcionales):
#   PYTHON, DATASET_NAME, TRAIN_DATA, VALID_DATA, RUN_NAME, SAVE_DIR, GEN_FRM_DIR,
#   IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, INPUT_LENGTH, TOTAL_LENGTH, NUM_HIDDEN,
#   FILTER_SIZE, PATCH_SIZE, BATCH_SIZE, LR, MAX_ITERATIONS, DISPLAY_INTERVAL,
#   TEST_INTERVAL, SNAPSHOT_INTERVAL, PRETRAINED
# Los args extra se pasan al final, asi que tienen prioridad sobre todo lo anterior.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[[ $# -ge 1 ]] || die "uso: $0 <base|dmref> [args extra]"
setup_project "$1"; shift

TRAIN_ARGS=(
    --is_training       1
    --lr                "${LR:-0.001}"
    --max_iterations    "${MAX_ITERATIONS:-80000}"
    --display_interval  "${DISPLAY_INTERVAL:-100}"
    --test_interval     "${TEST_INTERVAL:-5000}"
    --snapshot_interval "${SNAPSHOT_INTERVAL:-5000}"
)
if [[ -n "${PRETRAINED:-}" ]]; then
    TRAIN_ARGS+=(--pretrained_model "$(resolve_checkpoint "$PRETRAINED")")
fi

cd "$ROOT_DIR"
log "Entrenando proyecto '$PROJECT' (dataset: $DATASET_NAME)"
log "Checkpoints -> $SAVE_DIR"
run_logged "train_${RUN_NAME}" \
    "$PYTHON" "$TRAIN_PY" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
