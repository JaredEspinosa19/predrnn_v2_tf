import os
import datetime
import cv2
import numpy as np
from DMREF_predrnn_tf.core.utils import metrics
from skimage.metrics import structural_similarity as compare_ssim
from DMREF_predrnn_tf.core.utils import preprocess


def train(model, ims, real_input_flag, configs, itr):
    """
    Un paso de entrenamiento (con optional reverse_input).

    Args:
        model            : instancia de Model (model_factory)
        ims              : np.array [B, T, H, W, C]  en espacio de patches
        real_input_flag  : np.array [B, T-1, H, W, C] mascara de sampling
        configs          : objeto de configuracion
        itr              : iteracion actual (para logging)
    """
    cost = model.train(ims, real_input_flag)

    if configs.reverse_input:
        # Entrenar tambien con la secuencia invertida en el tiempo
        ims_rev = np.flip(ims, axis=1).copy()
        cost   += model.train(ims_rev, real_input_flag)
        cost   /= 2.0

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
              f'itr: {itr}  training loss: {cost:.6f}')


def test(model, test_input_handle, configs, itr):
    """
    Evaluacion completa sobre el conjunto de test.

    Metricas calculadas por frame de prediccion:
        - MSE
        - SSIM  (scikit-image)
        - PSNR

    Args:
        model             : instancia de Model
        test_input_handle : handle del data provider de test
        configs           : objeto de configuracion
        itr               : iteracion o etiqueta del checkpoint
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)

    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.makedirs(res_path, exist_ok=True)

    n_pred    = configs.total_length - configs.input_length
    avg_mse   = 0.0
    batch_id  = 0
    img_mse   = [0.0] * n_pred
    ssim_vals = [0.0] * n_pred
    psnr_vals = [0.0] * n_pred

    # Mascara de test: todos ceros (la red predice sin ver frames futuros)
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros((
        configs.batch_size,
        configs.total_length - mask_input - 1,
        configs.img_height // configs.patch_size,       # DMREF: H_patch = 412/4 = 103
        configs.img_width  // configs.patch_size,       # DMREF: W_patch = 1024/4 = 256
        configs.patch_size ** 2 * configs.img_channel   # DMREF: C_patch = 16
    ), dtype=np.float32)

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1] = 1.0

    while not test_input_handle.no_batch_left():
        batch_id += 1
        test_ims  = test_input_handle.get_batch()              # [B, T, H, W, C]
        test_dat  = preprocess.reshape_patch(test_ims, configs.patch_size)
        # TODO: [PASO 13] Verificar que configs.img_channel coincide con los canales reales
        #       de las imagenes DMREF. Si el dataset no tiene relleno, este slice es redundante
        #       pero inofensivo. Si tiene mas canales de los esperados, ajustar img_channel.
        test_ims  = test_ims[:, :, :, :, :configs.img_channel]   # IMAGE_SIZE: recorta al numero de canales reales (descarta relleno)

        img_gen   = model.test(test_dat, real_input_flag)      # [B, T-1, H, W, Cp]
        img_gen   = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        img_out   = img_gen[:, -n_pred:]                       # solo frames predichos

        # ---- Metricas por frame ----
        for i in range(n_pred):
            x  = test_ims[:, i + configs.input_length, :, :, :]   # ground truth
            gx = img_out[:, i, :, :, :]                            # prediccion
            gx = np.clip(gx, 0.0, 1.0)

            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse    += mse

            real_frm = np.uint8(x  * 255)
            pred_frm = np.uint8(gx * 255)

            psnr_vals[i] += metrics.batch_psnr(pred_frm, real_frm)

            for b in range(configs.batch_size):
                score, _ = compare_ssim(
                    pred_frm[b], real_frm[b],
                    full=True, multichannel=True
                )
                ssim_vals[i] += score

        # ---- Guardar ejemplos visuales ----
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.makedirs(path, exist_ok=True)
            for i in range(configs.total_length):
                name      = f'gt{i + 1}.png'
                img_gt    = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(os.path.join(path, name), img_gt)
            for i in range(n_pred):
                name   = f'pd{i + 1 + configs.input_length}.png'
                img_pd = np.clip(img_out[0, i], 0.0, 1.0)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(os.path.join(path, name), img_pd)

        test_input_handle.next()

    # ---- Resumen de metricas ----
    norm = batch_id * configs.batch_size
    avg_mse /= norm
    print(f'mse por secuencia: {avg_mse:.4f}')

    ssim_arr = np.array(ssim_vals, dtype=np.float32) / norm
    print(f'ssim promedio por frame: {np.mean(ssim_arr):.4f}')
    for v in ssim_arr:
        print(f'  {v:.4f}')

    psnr_arr = np.array(psnr_vals, dtype=np.float32) / batch_id
    print(f'psnr promedio por frame: {np.mean(psnr_arr):.4f}')
    for v in psnr_arr:
        print(f'  {v:.4f}')
