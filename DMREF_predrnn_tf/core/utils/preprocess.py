import numpy as np


def reshape_patch(img_tensor, patch_size):
    """
    Transforma el tensor de imagenes al espacio de patches.

    Entrada : [B, T, H, W, C]
    Salida  : [B, T, H/patch_size, W/patch_size, patch_size^2 * C]

    Identico a la version PyTorch (opera sobre numpy).
    """
    # TODO: [PASO 4] Esta funcion es generica y no requiere cambios para imagenes DMREF,
    #       SIEMPRE QUE H y W sean divisibles por patch_size.
    #       Si las imagenes DMREF no son cuadradas (H != W), la funcion ya lo soporta
    #       porque img_height e img_width se leen independientemente del tensor.
    #       Verificar que: img_height % patch_size == 0  y  img_width % patch_size == 0.
    assert img_tensor.ndim == 5
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape  # IMAGE_SIZE: H=img_height, W=img_width, C=num_channels

    a = np.reshape(img_tensor, [
        batch_size, seq_length,
        img_height // patch_size, patch_size,   # IMAGE_SIZE: divide H en H/patch_size bloques de patch_size filas
        img_width  // patch_size, patch_size,   # IMAGE_SIZE: divide W en W/patch_size bloques de patch_size columnas
        num_channels                            # IMAGE_SIZE: canales originales
    ])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    patch_tensor = np.reshape(b, [
        batch_size, seq_length,
        img_height // patch_size,               # IMAGE_SIZE: H_patch = H / patch_size
        img_width  // patch_size,               # IMAGE_SIZE: W_patch = W / patch_size
        patch_size * patch_size * num_channels  # IMAGE_SIZE: C_patch = patch_size^2 * C (pixeles del patch aplanados)
    ])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    """
    Invierte reshape_patch: vuelve del espacio de patches al espacio de imagen.

    Entrada : [B, T, H/patch_size, W/patch_size, patch_size^2 * C]
    Salida  : [B, T, H, W, C]

    Identico a la version PyTorch (opera sobre numpy).
    """
    # TODO: [PASO 13] Igual que reshape_patch, esta funcion es generica y soporta
    #       imagenes no cuadradas sin cambios. Verificar que la salida reconstruye
    #       correctamente las dimensiones originales de las imagenes DMREF.
    assert patch_tensor.ndim == 5
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape  # IMAGE_SIZE: H_patch, W_patch, C_patch
    img_channels = channels // (patch_size * patch_size)                               # IMAGE_SIZE: recupera C original = C_patch / patch_size^2

    a = np.reshape(patch_tensor, [
        batch_size, seq_length,
        patch_height, patch_width,   # IMAGE_SIZE: H_patch x W_patch
        patch_size, patch_size,      # IMAGE_SIZE: expande cada patch a su bloque de pixeles
        img_channels                 # IMAGE_SIZE: canales originales recuperados
    ])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [
        batch_size, seq_length,
        patch_height * patch_size,   # IMAGE_SIZE: H original = H_patch * patch_size
        patch_width  * patch_size,   # IMAGE_SIZE: W original = W_patch * patch_size
        img_channels                 # IMAGE_SIZE: C original
    ])
    return img_tensor
