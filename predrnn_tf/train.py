"""
PredRNN v2 - TensorFlow
=======================
Entry point de entrenamiento y evaluacion.

Uso rapido (equivalente al run.py de PyTorch):

    python train.py \
        --is_training 1 \
        --dataset_name mnist \
        --train_data_paths data/moving-mnist-example/moving-mnist-train.npz \
        --valid_data_paths data/moving-mnist-example/moving-mnist-valid.npz \
        --img_width 64 --img_channel 1 \
        --input_length 10 --total_length 20 \
        --num_hidden 64,64,64,64 \
        --filter_size 5 --patch_size 4 \
        --batch_size 8 --lr 0.001 \
        --max_iterations 80000

Para solo evaluar un checkpoint:

    python train.py \
        --is_training 0 \
        --pretrained_model checkpoints/mnist_predrnn_v2/model-5000 \
        ...mismos args de datos/modelo...
"""

import os
import sys
import math
import argparse
import numpy as np

# Asegurar que el directorio predrnn_tf/ sea el raiz de imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='PredRNN v2 - TensorFlow',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# entrenamiento / test
parser.add_argument('--is_training',   type=int,   default=1)

# datos
parser.add_argument('--dataset_name',      type=str, default='mnist')
parser.add_argument('--train_data_paths',  type=str,
                    default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths',  type=str,
                    default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir',          type=str, default='checkpoints/mnist_predrnn_v2_tf')
parser.add_argument('--gen_frm_dir',       type=str, default='results/mnist_predrnn_v2_tf')
parser.add_argument('--input_length',      type=int, default=10)
parser.add_argument('--total_length',      type=int, default=20)
parser.add_argument('--img_width',         type=int, default=64)   # IMAGE_SIZE: ancho/alto original de la imagen (cuadrada)
parser.add_argument('--img_channel',       type=int, default=1)    # IMAGE_SIZE: numero de canales (1=gris, 3=RGB)

# modelo
parser.add_argument('--model_name',        type=str,   default='predrnn_v2')
parser.add_argument('--pretrained_model',  type=str,   default='')
parser.add_argument('--num_hidden',        type=str,   default='64,64,64,64')
parser.add_argument('--filter_size',       type=int,   default=5)
parser.add_argument('--stride',            type=int,   default=1)
parser.add_argument('--patch_size',        type=int,   default=4)    # IMAGE_SIZE: factor de reduccion espacial (H_patch = img_width / patch_size)
parser.add_argument('--layer_norm',        type=int,   default=1)
parser.add_argument('--decouple_beta',     type=float, default=0.1)

# scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int,   default=0)
parser.add_argument('--r_sampling_step_1',          type=float, default=25000)
parser.add_argument('--r_sampling_step_2',          type=int,   default=50000)
parser.add_argument('--r_exp_alpha',                type=int,   default=5000)
parser.add_argument('--scheduled_sampling',         type=int,   default=1)
parser.add_argument('--sampling_stop_iter',         type=int,   default=50000)
parser.add_argument('--sampling_start_value',       type=float, default=1.0)
parser.add_argument('--sampling_changing_rate',     type=float, default=0.00002)

# optimizacion
parser.add_argument('--lr',                 type=float, default=0.001)
parser.add_argument('--reverse_input',      type=int,   default=1)
parser.add_argument('--batch_size',         type=int,   default=8)
parser.add_argument('--max_iterations',     type=int,   default=80000)
parser.add_argument('--display_interval',   type=int,   default=100)
parser.add_argument('--test_interval',      type=int,   default=5000)
parser.add_argument('--snapshot_interval',  type=int,   default=5000)
parser.add_argument('--num_save_samples',   type=int,   default=10)

args = parser.parse_args()
print(args)


# ---------------------------------------------------------------------------
# Funciones de Scheduled Sampling
# ---------------------------------------------------------------------------

def reserve_schedule_sampling_exp(itr):
    """Reverse scheduled sampling: empieza con predicciones, termina con reales."""
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(
            -float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) \
              * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    # Tokens aleatorios para frames de entrada y frames de prediccion
    r_random_flip = np.random.random_sample((args.batch_size, args.input_length - 1))
    r_true_token  = r_random_flip < r_eta

    random_flip   = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token    = random_flip < eta

    patch_shape = (
        args.img_width // args.patch_size,           # IMAGE_SIZE: H reducida tras el patch (ej: 64/4 = 16)
        args.img_width // args.patch_size,           # IMAGE_SIZE: W reducida tras el patch (ej: 64/4 = 16)
        args.patch_size ** 2 * args.img_channel,     # IMAGE_SIZE: C expandida del patch  (ej: 4*4*1 = 16)
    )
    ones  = np.ones(patch_shape)
    zeros = np.zeros(patch_shape)

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                real_input_flag.append(ones if r_true_token[i, j] else zeros)
            else:
                real_input_flag.append(
                    ones if true_token[i, j - (args.input_length - 1)] else zeros)

    real_input_flag = np.reshape(
        np.array(real_input_flag),
        (args.batch_size, args.total_length - 2, *patch_shape)
    )
    return real_input_flag


def schedule_sampling(eta, itr):
    """Scheduled sampling estandar: empieza con reales, mezcla gradualmente predicciones."""
    patch_shape = (
        args.img_width // args.patch_size,           # IMAGE_SIZE: H reducida tras el patch (ej: 64/4 = 16)
        args.img_width // args.patch_size,           # IMAGE_SIZE: W reducida tras el patch (ej: 64/4 = 16)
        args.patch_size ** 2 * args.img_channel,     # IMAGE_SIZE: C expandida del patch  (ej: 4*4*1 = 16)
    )
    zeros = np.zeros((args.batch_size, args.total_length - args.input_length - 1,
                      *patch_shape))

    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token  = random_flip < eta

    ones  = np.ones(patch_shape)
    zeros_patch = np.zeros(patch_shape)

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            real_input_flag.append(ones if true_token[i, j] else zeros_patch)

    real_input_flag = np.reshape(
        np.array(real_input_flag),
        (args.batch_size, args.total_length - args.input_length - 1, *patch_shape)
    )
    return eta, real_input_flag


# ---------------------------------------------------------------------------
# Bucle de entrenamiento
# ---------------------------------------------------------------------------

def train_wrapper(model):
    if args.pretrained_model:
        # Warm-up pass para inicializar variables antes de cargar pesos
        _dummy_warmup(model)
        model.load(args.pretrained_model)

    train_handle, test_handle = datasets_factory.data_provider(
        args.dataset_name,
        args.train_data_paths,
        args.valid_data_paths,
        args.batch_size,
        args.img_width,
        seq_length=args.total_length,
        is_training=True,
    )

    eta = args.sampling_start_value

    for itr in range(1, args.max_iterations + 1):
        if train_handle.no_batch_left():
            train_handle.begin(do_shuffle=True)

        ims = train_handle.get_batch()                              # [B, T, H, W, C]
        ims = preprocess.reshape_patch(ims, args.patch_size)       # [B, T, h, w, Cp]

        if args.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, ims, real_input_flag, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            trainer.test(model, test_handle, args, itr)

        train_handle.next()


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------

def test_wrapper(model):
    _dummy_warmup(model)
    model.load(args.pretrained_model)

    test_handle = datasets_factory.data_provider(
        args.dataset_name,
        args.train_data_paths,
        args.valid_data_paths,
        args.batch_size,
        args.img_width,
        seq_length=args.total_length,
        is_training=False,
    )
    trainer.test(model, test_handle, args, 'test_result')


def _dummy_warmup(model):
    """
    Ejecuta un forward pass con datos dummy para inicializar las variables
    de TF antes de cargar pesos desde un checkpoint.
    """
    ph = args.img_width // args.patch_size           # IMAGE_SIZE: dimension espacial reducida (H = W = img_width / patch_size)
    cp = args.patch_size ** 2 * args.img_channel     # IMAGE_SIZE: canales expandidos del patch
    dummy_frames = np.zeros(
        (args.batch_size, args.total_length, ph, ph, cp), dtype=np.float32)      # IMAGE_SIZE: [B, T, H_patch, W_patch, C_patch]
    dummy_mask   = np.zeros(
        (args.batch_size, args.total_length - args.input_length - 1, ph, ph, cp),  # IMAGE_SIZE: mismas dims espaciales que frames
        dtype=np.float32)
    model.test(dummy_frames, dummy_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs(args.save_dir,    exist_ok=True)
os.makedirs(args.gen_frm_dir, exist_ok=True)

print('Inicializando modelo...')
model = Model(args)

if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
