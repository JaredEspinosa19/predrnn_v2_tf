#!/usr/bin/env bash
# Funciones y valores compartidos por train.sh y test.sh.
# No se ejecuta directamente: se carga con `source`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

# Los imports de DMREF_predrnn_tf son absolutos (DMREF_predrnn_tf.core...),
# asi que la raiz del repo debe estar en PYTHONPATH.
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Selecciona el proyecto y define sus argumentos por defecto.
# Cada valor se puede sobreescribir con una variable de entorno del mismo nombre.
#   base  -> predrnn_tf/        (MNIST/KTH, imagenes cuadradas)
#   dmref -> DMREF_predrnn_tf/  (imagenes 1024x412)
setup_project() {
    PROJECT="$1"
    case "$PROJECT" in
        base)
            TRAIN_PY="$ROOT_DIR/predrnn_tf/train.py"
            DATASET_NAME="${DATASET_NAME:-mnist}"
            TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/data/moving-mnist-example/moving-mnist-train.npz}"
            VALID_DATA="${VALID_DATA:-$ROOT_DIR/data/moving-mnist-example/moving-mnist-valid.npz}"
            IMG_ARGS=(--img_width "${IMG_WIDTH:-64}")
            ;;
        dmref)
            TRAIN_PY="$ROOT_DIR/DMREF_predrnn_tf/train.py"
            # 'dmref' debe registrarse en datasets_factory.py (ver TODO PASO 2).
            DATASET_NAME="${DATASET_NAME:-dmref}"
            TRAIN_DATA="${TRAIN_DATA:-$ROOT_DIR/DMREF_dataset/ground_truth}"
            VALID_DATA="${VALID_DATA:-$ROOT_DIR/DMREF_dataset/ground_truth}"
            IMG_ARGS=(--img_width "${IMG_WIDTH:-1024}" --img_height "${IMG_HEIGHT:-412}")
            ;;
        *)
            die "proyecto desconocido '$PROJECT' (usa: base | dmref)"
            ;;
    esac

    RUN_NAME="${RUN_NAME:-${PROJECT}_${DATASET_NAME}_predrnn_v2}"
    SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"
    GEN_FRM_DIR="${GEN_FRM_DIR:-$ROOT_DIR/results/$RUN_NAME}"

    COMMON_ARGS=(
        --dataset_name     "$DATASET_NAME"
        --train_data_paths "$TRAIN_DATA"
        --valid_data_paths "$VALID_DATA"
        --save_dir         "$SAVE_DIR"
        --gen_frm_dir      "$GEN_FRM_DIR"
        "${IMG_ARGS[@]}"
        --img_channel      "${IMG_CHANNEL:-1}"
        --input_length     "${INPUT_LENGTH:-10}"
        --total_length     "${TOTAL_LENGTH:-20}"
        --num_hidden       "${NUM_HIDDEN:-64,64,64,64}"
        --filter_size      "${FILTER_SIZE:-5}"
        --patch_size       "${PATCH_SIZE:-4}"
        --batch_size       "${BATCH_SIZE:-8}"
    )
}

# Resuelve un checkpoint: si recibe un directorio, devuelve el model-N mas reciente.
# Acepta checkpoints TF (model-N.index) y Keras 3 (model-N.weights.h5).
resolve_checkpoint() {
    local target="$1"
    if [[ -d "$target" ]]; then
        local latest
        latest="$(ls "$target" 2>/dev/null \
            | grep -E '^model-[0-9]+(\.index|\.weights\.h5)$' \
            | sed -E 's/(\.index|\.weights\.h5)$//' \
            | sort -t- -k2 -n | tail -n 1)"
        [[ -n "$latest" ]] || die "no hay checkpoints model-N en $target"
        if [[ -f "$target/$latest.weights.h5" ]]; then
            echo "$target/$latest.weights.h5"
        else
            echo "$target/$latest"
        fi
    else
        echo "$target"
    fi
}

# Ejecuta el comando mostrando la salida y guardandola en logs/.
run_logged() {
    local tag="$1"; shift
    mkdir -p "$LOG_DIR"
    local log_file="$LOG_DIR/${tag}_$(date '+%Y%m%d_%H%M%S').log"
    log "Log: $log_file"
    log "Comando: $*"
    "$@" 2>&1 | tee "$log_file"
}
