# TODOs — Adaptación de dimensiones de imagen para dataset DMREF

Referencia de cambios ya realizados: ver `CAMBIOS_REALIZADOS.md`.

---

## PASO 1 — Parámetros de entrada ✅ RESUELTO
**Archivo:** `train.py`

### Qué se hizo
Se definieron los tres parámetros que controlan las dimensiones en todo el pipeline.
Las imágenes DMREF son **1024×413, escala de grises**, pero `GCD(413, 1024) = 1`,
por lo que no existe `patch_size > 1` que divida exactamente ambas dimensiones.
Se adoptó recortar 1 píxel de alto → **412×1024** con `patch_size=4`.

```python
--img_width   = 1024   # ancho original DMREF
--img_height  = 412    # alto recortado (413 → 412, divisible por 4)
--img_channel = 1      # escala de grises
--patch_size  = 4      # H_patch=103, W_patch=256, C_patch=16
```

### Por qué es necesario
Estos parámetros son la fuente de verdad de la que depende todo el pipeline.
Sin valores correctos aquí, las máscaras de sampling, los tensores dummy del warmup
y el data provider construirían arrays con shapes incompatibles entre sí, provocando
errores de dimensiones en tiempo de ejecución.
El parámetro `img_height` es nuevo (no existía en la versión MNIST) porque las imágenes
DMREF son rectangulares y ya no se puede asumir que alto = ancho.

Otras opciones de `patch_size` disponibles:
| patch_size | recorte de alto | H_patch | W_patch | C_patch |
|-----------|----------------|---------|---------|---------|
| 4         | 412            | 103     | 256     | 16      |
| 8         | 408            | 51      | 128     | 64      |
| 16        | 400            | 25      | 64      | 256     |

---

## PASO 2 — Crear el data provider del dataset DMREF
**Archivos a crear/modificar:**
- Nuevo archivo: `core/data_provider/dmref_dataset.py`
- Modificar: `core/data_provider/datasets_factory.py`

### Qué se debe hacer

Crear el módulo `dmref_dataset.py` con una clase `InputHandle` que tenga exactamente
la misma interfaz que `mnist.py`. Esa clase es la que `datasets_factory.py` llama para
obtener los batches de datos, de modo que el bucle de entrenamiento en `train.py` no
necesita saber nada del formato específico de las imágenes DMREF.

La clase debe implementar estos cuatro métodos:

| Método | Qué hace |
|--------|---------|
| `begin(do_shuffle)` | Reinicia el iterador al inicio del dataset; si `do_shuffle=True`, baraja el orden de las muestras |
| `get_batch()` | Retorna el batch actual como array numpy `[B, T, 412, 1024, 1]` en float32, valores en [0,1] |
| `next()` | Avanza el puntero interno al siguiente batch |
| `no_batch_left()` | Retorna `True` cuando se agotaron todos los batches del epoch |

Pasos concretos dentro del proveedor:
1. **Leer las imágenes** del disco en el formato que tenga el dataset (carpeta de archivos, `.npz`, `.h5`, etc.)
2. **Recortar 1 fila de alto**: `imagen[:412, :]` para ir de 413 → 412 píxeles
3. **Normalizar** a rango [0, 1]: `imagen / max_valor` (para uint8 sería `/ 255`, para uint16 `/ 65535`)
4. **Agregar dimensión de canal**: `imagen[:, :, np.newaxis]` para obtener shape `[412, 1024, 1]`
5. **Agrupar en secuencias** de longitud `total_length` (ej: 20 frames consecutivos por muestra)
6. **Empaquetar en batches** de tamaño `batch_size`

En `datasets_factory.py` agregar:
```python
from DMREF_predrnn_tf.core.data_provider import dmref_dataset

datasets_map = {
    'mnist': mnist,
    'dmref': dmref_dataset,   # <-- agregar
}
```

### Por qué es necesario
Este es el único eslabón pendiente para poder correr el entrenamiento.
El data provider es la puerta de entrada del dataset: sin él, el modelo no tiene
forma de leer las imágenes DMREF. Todo el resto del pipeline (reshape_patch, la red,
las métricas, el guardado de resultados) ya está adaptado y espera recibir tensores
`[B, T, 412, 1024, 1]` como punto de partida.

---

## PASO 4 — `reshape_patch`: imagen → espacio de patches ✅ SIN CAMBIOS
**Archivo:** `core/utils/preprocess.py`

### Qué hace
Transforma cada frame de `[H, W, C]` al espacio de patches `[H/p, W/p, p²·C]`.
Divide la imagen en bloques de `patch_size × patch_size` píxeles y los aplana
en la dimensión de canales. Esto reduce la resolución espacial y aumenta los canales,
lo que permite que las convoluciones de la red operen sobre un espacio más compacto.

### Por qué no requiere cambios
La función lee `img_height` e `img_width` directamente del tensor de entrada,
por lo que ya soporta imágenes no cuadradas sin modificación.
Con las imágenes DMREF: `[B, T, 412, 1024, 1]` → `[B, T, 103, 256, 16]`.

Verificación previa obligatoria: `412 % 4 == 0` ✅ y `1024 % 4 == 0` ✅.

---

## PASO 5 — Máscaras de scheduled sampling y warmup ✅ RESUELTO
**Archivos:** `train.py`, `core/trainer.py`

### Qué se hizo
Se reemplazó el uso de `img_width` como sustituto del alto por el nuevo parámetro
`img_height`, tanto en `patch_shape` (funciones de sampling en `train.py`) como en
`real_input_flag` (función `test` en `trainer.py`).

### Por qué es necesario
Las máscaras de scheduled sampling son tensores que se mezclan frame a frame con
los datos reales durante el entrenamiento. Si su shape espacial no coincide exactamente
con el shape de los frames de entrada, TensorFlow lanza un error de broadcast.
Con imágenes cuadradas (MNIST), `H_patch == W_patch` y usar `img_width` para ambas
dimensiones funcionaba; con DMREF (`H_patch=103 ≠ W_patch=256`) ya no.

```python
# Resultado correcto para DMREF:
patch_shape = (103, 256, 16)   # (H_patch, W_patch, C_patch)
```

---

## PASO 6 — `frame_channel` en el modelo ✅ SIN CAMBIOS
**Archivo:** `core/models/predrnn_v2.py`

### Qué hace
`frame_channel` es el número de canales que tiene cada frame una vez transformado al
espacio de patches. Es el valor que determina el tamaño de la primera convolución
de la celda ST-LSTM (la que procesa la entrada `x_t`).

### Por qué no requiere cambios
Se recalcula automáticamente a partir de los parámetros ya actualizados:
```python
frame_channel = patch_size² × img_channel = 4² × 1 = 16
```
Cualquier cambio en `patch_size` o `img_channel` se propaga aquí sin tocar este archivo.

---

## PASO 7 — Dimensiones espaciales en el forward pass ✅ SIN CAMBIOS
**Archivo:** `core/models/predrnn_v2.py`

### Qué hace
Al inicio de cada forward pass, el modelo lee el alto y ancho del tensor de entrada
con `tf.shape()`. Esos valores se usan para inicializar los estados ocultos de cada
capa con las dimensiones correctas.

### Por qué no requiere cambios
`tf.shape()` lee el shape real del tensor en tiempo de ejecución, no en tiempo de
construcción del grafo. Por eso el modelo se adapta automáticamente a cualquier
resolución que llegue, sin importar si es cuadrada o no.
Con DMREF: `height=103`, `width=256` (en espacio de patches).

---

## PASO 8 — Inicialización de estados ocultos ✅ SIN CAMBIOS
**Archivo:** `core/models/predrnn_v2.py`

### Qué hace
Al comienzo de cada secuencia, los estados internos de la red (`h`, `c`, `delta_c`,
`delta_m` por capa, y la memoria compartida `M`) se inicializan a cero con el shape
`[B, H_patch, W_patch, num_hidden]`.

### Por qué no requiere cambios
Los valores de `H_patch` y `W_patch` vienen del PASO 7, que ya es dinámico.
El único parámetro que podría necesitar ajuste es `num_hidden`, que controla
la capacidad de la red (no las dimensiones de imagen). Por defecto es `64,64,64,64`
y puede modificarse en `train.py` según los recursos disponibles.

---

## PASO 9 — LayerNormalization en la celda ST-LSTM ✅ SIN CAMBIOS
**Archivo:** `core/layers/SpatioTemporalLSTMCell_v2.py`

### Qué hace
Aplica normalización de capa sobre los tensores intermedios dentro de cada celda
ST-LSTM para estabilizar el entrenamiento. Se aplica después de cada convolución
principal (`conv_x`, `conv_h`, `conv_m`, `conv_o`).

### Por qué no requiere cambios
`axis=[1,2,3]` normaliza sobre todos los ejes espaciales y de canal `[H, W, C]`,
lo que es correcto para cualquier resolución en formato channels-last.
Los parámetros aprendibles `gamma` y `beta` se crean automáticamente con el shape
correcto al ejecutarse el primer forward pass, sin importar el tamaño de la imagen.

---

## PASO 13 — `reshape_patch_back`: patches → imagen ✅ SIN CAMBIOS
**Archivo:** `core/utils/preprocess.py`

### Qué hace
Operación inversa a `reshape_patch`. Toma las predicciones de la red en el espacio
de patches `[B, T-1, H_patch, W_patch, C_patch]` y las reconstruye al espacio de
imagen original `[B, T-1, H, W, C]`. Esta función se llama en `trainer.py` justo
antes de calcular las métricas y guardar los frames predichos.

### Por qué no requiere cambios
Al igual que `reshape_patch`, lee las dimensiones directamente del tensor de entrada,
por lo que ya soporta imágenes no cuadradas.
Con DMREF: `[B, T-1, 103, 256, 16]` → `[B, T-1, 412, 1024, 1]`.

---

## Resumen de estado

| Paso | Archivo | Qué se debe hacer | Por qué | Estado |
|------|---------|-------------------|---------|--------|
| 1 | `train.py` | Definir `img_width=1024`, `img_height=412`, `img_channel=1`, `patch_size=4` | Son la fuente de verdad de todo el pipeline | ✅ Hecho |
| 2 | `dmref_dataset.py` (nuevo) | Crear clase `InputHandle` que lea, recorte, normalice y sirva batches `[B,T,412,1024,1]` | Sin esto la red no puede leer las imágenes DMREF | **Pendiente** |
| 2 | `datasets_factory.py` | Registrar `dmref_dataset` en el mapa de datasets | Para poder usar `--dataset_name dmref` al lanzar el entrenamiento | **Pendiente** |
| 4 | `preprocess.py` | Ninguno | Lee dims del tensor; ya soporta H≠W | ✅ OK |
| 5 | `train.py`, `trainer.py` | Separar `img_height` e `img_width` en `patch_shape` y `real_input_flag` | Las máscaras deben tener el mismo shape que los frames de entrada | ✅ Hecho |
| 6 | `predrnn_v2.py` | Ninguno | `frame_channel` se recalcula solo desde PASO 1 | ✅ OK |
| 7 | `predrnn_v2.py` | Ninguno | `tf.shape()` es dinámico en tiempo de ejecución | ✅ OK |
| 8 | `predrnn_v2.py` | Ninguno | Los estados usan las dims dinámicas del PASO 7 | ✅ OK |
| 9 | `SpatioTemporalLSTMCell_v2.py` | Ninguno | `axis=[1,2,3]` es genérico para cualquier resolución | ✅ OK |
| 13 | `preprocess.py` | Ninguno | Lee dims del tensor; ya soporta H≠W | ✅ OK |
