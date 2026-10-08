#!/usr/bin/env bash
# Evalua un checkpoint de PredRNN v2 sobre el conjunto de validacion.
#
# Uso:
#   scripts/test.sh <base|dmref> [checkpoint|directorio] [args extra para train.py]
#
# Si no se indica checkpoint se usa el mas reciente de checkpoints/<RUN_NAME>/.
# Si se pasa un directorio se usa el model-N con mayor N dentro de el.
#
# Ejemplos:
#   scripts/test.sh base
#   scripts/test.sh dmref checkpoints/dmref_dmref_predrnn_v2/model-20000
#   scripts/test.sh dmref "" --num_save_samples 20
#
# Usa las mismas variables de entorno que train.sh para datos y modelo:
# deben coincidir con las del entrenamiento para que los pesos carguen.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[[ $# -ge 1 ]] || die "uso: $0 <base|dmref> [checkpoint|directorio] [args extra]"
setup_project "$1"; shift

CKPT_ARG="${1:-}"
[[ $# -ge 1 ]] && shift
CHECKPOINT="$(resolve_checkpoint "${CKPT_ARG:-$SAVE_DIR}")"

GEN_FRM_DIR="${GEN_FRM_DIR}_test"
COMMON_ARGS+=(--gen_frm_dir "$GEN_FRM_DIR")

cd "$ROOT_DIR"
log "Evaluando proyecto '$PROJECT' (dataset: $DATASET_NAME)"
log "Checkpoint: $CHECKPOINT"
log "Resultados -> $GEN_FRM_DIR/test_result"
run_logged "test_${RUN_NAME}" \
    "$PYTHON" "$TRAIN_PY" "${COMMON_ARGS[@]}" \
    --is_training 0 \
    --pretrained_model "$CHECKPOINT" \
    "$@"
