import tensorflow as tf
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell


class RNN(tf.keras.Model):
    """
    PredRNN v2 - TensorFlow implementation.

    Equivalente a core/models/predrnn_v2.py en PyTorch.

    El modelo trabaja internamente en formato channels-last [B, H, W, C],
    por lo que NO se requiere permutar el tensor de entrada como en PyTorch.

    Entradas:
        frames_tensor : np.array o tf.Tensor  [B, T, H, W, C]
                        (en espacio de patches, tras reshape_patch)
        mask_true     : np.array o tf.Tensor  [B, T-1, H, W, C]
                        (mascara de scheduled sampling)

    Salidas:
        next_frames : tf.Tensor [B, T-1, H, W, C]
        loss        : escalar   MSE + decouple_beta * decouple_loss
    """

    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        self.configs       = configs
        self.num_layers    = num_layers
        self.num_hidden    = num_hidden

        # Canales de cada frame en el espacio de patches
        # patch_size^2 * img_channel  (ej: 4*4*1 = 16)
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel  # IMAGE_SIZE: C_patch = patch_size^2 * img_channel

        # Construir lista de celdas ST-LSTM
        self.cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell = SpatioTemporalLSTMCell(
                in_channel, num_hidden[i],
                configs.filter_size, configs.stride, configs.layer_norm,
                name=f'st_lstm_{i}'
            )
            self.cell_list.append(cell)
            # Registrar como atributo para que Keras rastree los pesos
            setattr(self, f'cell_{i}', cell)

        # Proyeccion final: num_hidden[-1] -> frame_channel  (conv 1x1)
        self.conv_last = tf.keras.layers.Conv2D(
            self.frame_channel, 1, strides=1, padding='same',
            use_bias=False, name='conv_last'
        )

        # Adaptador compartido para el decouple loss: num_hidden[0] -> num_hidden[0]  (conv 1x1)
        self.adapter = tf.keras.layers.Conv2D(
            num_hidden[0], 1, strides=1, padding='same',
            use_bias=False, name='adapter'
        )

    def call(self, frames_tensor, mask_true, training=False):
        """
        Args:
            frames_tensor : [B, T, H, W, C]   (float32)
            mask_true     : [B, T-1, H, W, C] (float32, valores 0 o 1)
        Returns:
            next_frames   : [B, T-1, H, W, C]
            loss          : escalar
        """
        frames_tensor = tf.cast(frames_tensor, tf.float32)
        mask_true     = tf.cast(mask_true,     tf.float32)

        batch  = tf.shape(frames_tensor)[0]
        height = tf.shape(frames_tensor)[2]   # IMAGE_SIZE: H_patch = img_width // patch_size
        width  = tf.shape(frames_tensor)[3]   # IMAGE_SIZE: W_patch = img_width // patch_size

        # Inicializar estados ocultos h, c, delta_c, delta_m por capa
        h_t          = []
        c_t          = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = tf.zeros([batch, height, width, self.num_hidden[i]])  # IMAGE_SIZE: estados h/c iniciales [B, H_patch, W_patch, num_hidden]
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # Memoria M compartida entre capas (mismas dims que h de capa 0)
        memory = tf.zeros([batch, height, width, self.num_hidden[0]])   # IMAGE_SIZE: memoria M [B, H_patch, W_patch, num_hidden[0]]

        next_frames   = []
        decouple_loss = []
        x_gen         = None   # Se define en t=0 o t=input_length

        for t in range(self.configs.total_length - 1):

            # ---- Scheduled Sampling ----
            if self.configs.reverse_scheduled_sampling == 1:
                # Reverse: al inicio usa prediccion, luego real
                if t == 0:
                    net = frames_tensor[:, t]
                else:
                    net = (mask_true[:, t - 1] * frames_tensor[:, t]
                           + (1.0 - mask_true[:, t - 1]) * x_gen)
            else:
                # Forward: al inicio usa real, luego va mezclando
                if t < self.configs.input_length:
                    net = frames_tensor[:, t]
                else:
                    idx = t - self.configs.input_length
                    net = (mask_true[:, idx] * frames_tensor[:, t]
                           + (1.0 - mask_true[:, idx]) * x_gen)

            # ---- Paso por todas las capas ST-LSTM ----
            # Capa 0: entrada = net (frame actual o mezcla)
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                net, h_t[0], c_t[0], memory
            )
            delta_c_list[0] = self._normalize_via_adapter(delta_c)
            delta_m_list[0] = self._normalize_via_adapter(delta_m)

            # Capas 1..N-1: entrada = h de la capa anterior
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
                delta_c_list[i] = self._normalize_via_adapter(delta_c)
                delta_m_list[i] = self._normalize_via_adapter(delta_m)

            # Proyeccion al espacio de frames
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

            # ---- Decouple loss por cada capa en este paso de tiempo ----
            for i in range(self.num_layers):
                cos_sim = self._cosine_similarity(delta_c_list[i], delta_m_list[i])
                decouple_loss.append(tf.reduce_mean(tf.abs(cos_sim)))

        # Apilar predicciones: lista de [B, H, W, C] -> [B, T-1, H, W, C]
        next_frames = tf.stack(next_frames, axis=1)

        # Loss total = MSE + beta * decouple
        mse_loss      = tf.reduce_mean(tf.square(next_frames - frames_tensor[:, 1:]))
        decouple_loss = tf.reduce_mean(tf.stack(decouple_loss))
        loss          = mse_loss + self.configs.decouple_beta * decouple_loss

        return next_frames, loss

    def _normalize_via_adapter(self, delta):
        """
        Aplica el adaptador 1x1 y normaliza L2 a lo largo de la dimension espacial.

        Equivalente PyTorch:
            F.normalize(self.adapter(delta).view(B, C, -1), dim=2)

        delta : [B, H, W, C]
        return: [B, C, H*W]  normalizado por filas
        """
        x = self.adapter(delta)                       # [B, H, W, C]           # IMAGE_SIZE: H_patch x W_patch
        # Transponer a channels-first para replicar el reshape de PyTorch
        x = tf.transpose(x, perm=[0, 3, 1, 2])        # [B, C, H, W]          # IMAGE_SIZE: reordena dims espaciales
        b = tf.shape(x)[0]
        c = tf.shape(x)[1]
        x = tf.reshape(x, [b, c, -1])                 # [B, C, H*W]            # IMAGE_SIZE: aplana la dimension espacial H_patch * W_patch
        return tf.math.l2_normalize(x, axis=2)        # normalizar sobre H*W

    def _cosine_similarity(self, a, b):
        """
        Similitud coseno entre dos tensores ya normalizados L2.

        a, b : [B, C, H*W]
        return: [B, C]
        """
        return tf.reduce_sum(a * b, axis=2)
