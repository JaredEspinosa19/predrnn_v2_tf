# Cambios realizados — Adaptación PredRNN para dataset DMREF

Dataset objetivo: imágenes **1024×413, escala de grises**.

---

## Problema de divisibilidad

Las imágenes DMREF miden 1024×413 píxeles. La arquitectura PredRNN requiere
que ambas dimensiones sean divisibles exactamente por `patch_size`, porque la
operación `reshape_patch` divide H y W en bloques de `patch_size×patch_size`.

```
GCD(413, 1024) = 1  →  no existe patch_size > 1 que divida ambas dimensiones
```

**Solución adoptada:** recortar 1 píxel de alto → **412×1024**, `patch_size=4`.
Pérdida de información: 1/413 ≈ 0.24% de la altura.

Dimensiones resultantes en el espacio de patches:
```
H_patch = 412 / 4 = 103
W_patch = 1024 / 4 = 256
C_patch = 4² × 1  = 16
```

Otras opciones disponibles si se cambia de criterio:

| patch_size | recorte de alto | H_patch | W_patch | C_patch | píxeles perdidos |
|-----------|----------------|---------|---------|---------|-----------------|
| 4         | 412            | 103     | 256     | 16      | 1 fila (0.24%)  |
| 8         | 408            | 51      | 128     | 64      | 5 filas (1.21%) |
| 16        | 400            | 25      | 64      | 256     | 13 filas (3.1%) |

---

## Cambio 1 — Parámetros de imagen en `train.py`

**Archivo:** `train.py` — sección de argumentos de datos

**Qué se cambió:**

Antes (configuración para MNIST):
```python
parser.add_argument('--img_width',   type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--patch_size',  type=int, default=4)
# (no existia --img_height, se asumia imagen cuadrada)
```

Después (configuración para DMREF):
```python
parser.add_argument('--img_width',   type=int, default=1024)  # ancho DMREF
parser.add_argument('--img_height',  type=int, default=412)   # alto recortado
parser.add_argument('--img_channel', type=int, default=1)     # escala de grises
parser.add_argument('--patch_size',  type=int, default=4)     # sin cambio
```

**Por qué:** Los valores originales eran para MNIST (imágenes 64×64). Las imágenes
DMREF son rectangulares (1024×413), por lo que se necesita un parámetro separado
`img_height` para que el resto del código pueda distinguir alto y ancho.

---

## Cambio 2 — `patch_shape` en `reserve_schedule_sampling_exp` (`train.py`)

**Archivo:** `train.py` — función `reserve_schedule_sampling_exp`

**Qué se cambió:**

Antes (asumía imagen cuadrada usando `img_width` para ambas dimensiones):
```python
patch_shape = (
    args.img_width // args.patch_size,   # usaba img_width como H
    args.img_width // args.patch_size,   # usaba img_width como W
    args.patch_size ** 2 * args.img_channel,
)
```

Después (usa `img_height` e `img_width` por separado):
```python
patch_shape = (
    args.img_height // args.patch_size,  # H_patch = 412/4 = 103
    args.img_width  // args.patch_size,  # W_patch = 1024/4 = 256
    args.patch_size ** 2 * args.img_channel,
)
```

**Por qué:** `patch_shape` define el shape de la máscara de scheduled sampling.
Si se usara `img_width` para ambas dimensiones con las imágenes DMREF, la máscara
tendría shape `(256, 256, 16)` en lugar del correcto `(103, 256, 16)`, causando
un error de shape cuando se mezcla con los frames reales durante el entrenamiento.

---

## Cambio 3 — `patch_shape` en `schedule_sampling` (`train.py`)

**Archivo:** `train.py` — función `schedule_sampling`

**Qué se cambió:** Mismo cambio que en el Cambio 2. La función `schedule_sampling`
tiene su propia copia de `patch_shape` con el mismo problema.

```python
# Antes:
patch_shape = (args.img_width // ..., args.img_width // ..., ...)

# Después:
patch_shape = (args.img_height // ..., args.img_width // ..., ...)
```

**Por qué:** Igual que el Cambio 2. Esta función es el camino de scheduled sampling
estándar (sin reverse), y construye la misma máscara con el mismo shape incorrecto
si no se corrige.

---

## Cambio 4 — `_dummy_warmup` (`train.py`)

**Archivo:** `train.py` — función `_dummy_warmup`

**Qué se cambió:**

Antes (una sola variable `ph` para H y W):
```python
ph = args.img_width // args.patch_size
dummy_frames = np.zeros((args.batch_size, args.total_length, ph, ph, cp))
dummy_mask   = np.zeros((args.batch_size, ..., ph, ph, cp))
```

Después (variables separadas `ph_h` y `ph_w`):
```python
ph_h = args.img_height // args.patch_size   # 103
ph_w = args.img_width  // args.patch_size   # 256
dummy_frames = np.zeros((args.batch_size, args.total_length, ph_h, ph_w, cp))
dummy_mask   = np.zeros((args.batch_size, ..., ph_h, ph_w, cp))
```

**Por qué:** `_dummy_warmup` ejecuta un forward pass con datos ficticios para
inicializar los pesos de TensorFlow antes de cargar un checkpoint. Si los datos
dummy tienen el shape incorrecto (cuadrado en vez de rectangular), TF inicializa
los pesos con las dimensiones equivocadas y la carga del checkpoint falla.

---

## Cambio 5 — `real_input_flag` en `core/trainer.py`

**Archivo:** `core/trainer.py` — función `test`

**Qué se cambió:**

Antes (asumía imagen cuadrada):
```python
real_input_flag = np.zeros((
    configs.batch_size,
    configs.total_length - mask_input - 1,
    configs.img_width  // configs.patch_size,   # incorrecto para H
    configs.img_width  // configs.patch_size,
    configs.patch_size ** 2 * configs.img_channel
), dtype=np.float32)
```

Después:
```python
real_input_flag = np.zeros((
    configs.batch_size,
    configs.total_length - mask_input - 1,
    configs.img_height // configs.patch_size,   # H_patch = 103
    configs.img_width  // configs.patch_size,   # W_patch = 256
    configs.patch_size ** 2 * configs.img_channel
), dtype=np.float32)
```

**Por qué:** `real_input_flag` es la máscara de scheduled sampling durante la
evaluación. En test, es un tensor de ceros (la red predice sin ver frames futuros).
Si tiene el shape incorrecto no coincide con los frames de entrada y el modelo
lanza un error de dimensiones al ejecutar `model.test()`.

---

## Qué NO se cambió (y por qué no es necesario)

| Componente | Motivo |
|-----------|--------|
| `core/utils/preprocess.py` — `reshape_patch` | Lee `img_height` e `img_width` directamente del tensor; ya soporta H≠W |
| `core/utils/preprocess.py` — `reshape_patch_back` | Ídem, reconstruye correctamente cualquier dimensión |
| `core/models/predrnn_v2.py` — `frame_channel` | Se recalcula como `patch_size² × img_channel`; solo depende de los parámetros ya cambiados |
| `core/models/predrnn_v2.py` — estados h, c, M | Se inicializan con `tf.shape()` en tiempo de ejecución; son completamente dinámicos |
| `core/layers/SpatioTemporalLSTMCell_v2.py` | Las convoluciones tienen `padding='same'`; operan sobre cualquier tamaño espacial. `LayerNorm` con `axis=[1,2,3]` es genérico |
| `core/models/model_factory.py` | Solo gestiona el optimizador y los gradientes; no toca dimensiones |

---

## Flujo de dimensiones con las imágenes DMREF

```
Imagen original DMREF
    [B, T, 413, 1024, 1]
        ↓ recorte de 1 fila (en el data provider)
    [B, T, 412, 1024, 1]
        ↓ reshape_patch (patch_size=4)
    [B, T, 103, 256, 16]     ← toda la red opera aquí
        ↓ ST-LSTM × num_layers (convs sobre 103×256)
    [B, T-1, 103, 256, 16]
        ↓ reshape_patch_back
    [B, T-1, 412, 1024, 1]   ← predicciones reconstruidas
```
