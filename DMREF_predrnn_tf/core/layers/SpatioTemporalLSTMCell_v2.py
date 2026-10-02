import tensorflow as tf


class SpatioTemporalLSTMCell(tf.keras.layers.Layer):
    """
    SpatioTemporal LSTM Cell v2 - TensorFlow implementation.

    Equivalente a la version PyTorch en:
        core/layers/SpatioTemporalLSTMCell_v2.py

    Diferencias tecnicas con PyTorch:
      - PyTorch usa formato [B, C, H, W] (channels-first).
      - TensorFlow usa formato [B, H, W, C] (channels-last).
      - El LayerNorm se aplica sobre los ejes [H, W, C] (equivalente
        a normalizar sobre [C, H, W] en PyTorch, mismo conjunto de elementos).

    Entradas:
        x_t  : [B, H, W, in_channel]
        h_t  : [B, H, W, num_hidden]
        c_t  : [B, H, W, num_hidden]
        m_t  : [B, H, W, num_hidden]

    Salidas:
        h_new, c_new, m_new : [B, H, W, num_hidden]
        delta_c, delta_m    : [B, H, W, num_hidden]  (para decouple loss)
    """

    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden = num_hidden
        self._forget_bias = 1.0
        self.use_layer_norm = layer_norm

        # conv_x : in_channel -> num_hidden * 7
        self.conv_x = tf.keras.layers.Conv2D(
            num_hidden * 7, filter_size, strides=stride,
            padding='same', use_bias=False
        )
        # conv_h : num_hidden -> num_hidden * 4
        self.conv_h = tf.keras.layers.Conv2D(
            num_hidden * 4, filter_size, strides=stride,
            padding='same', use_bias=False
        )
        # conv_m : num_hidden -> num_hidden * 3
        self.conv_m = tf.keras.layers.Conv2D(
            num_hidden * 3, filter_size, strides=stride,
            padding='same', use_bias=False
        )
        # conv_o : num_hidden * 2 -> num_hidden
        self.conv_o = tf.keras.layers.Conv2D(
            num_hidden, filter_size, strides=stride,
            padding='same', use_bias=False
        )
        # conv_last : num_hidden * 2 -> num_hidden  (1x1)
        self.conv_last = tf.keras.layers.Conv2D(
            num_hidden, 1, strides=1, padding='same', use_bias=False
        )

        if layer_norm:
            # Normalizar sobre [H, W, C] = ejes 1, 2, 3 (equivalente a PyTorch [C, H, W])
            # axis=[1,2,3] => normaliza sobre toda la dimension espacial H_patch x W_patch x C
            # TODO: [PASO 9] axis=[1,2,3] es correcto para cualquier resolucion espacial en
            #       formato channels-last [B, H, W, C]. No requiere cambios para imagenes DMREF
            #       de distinto tamano. Los parametros gamma/beta de LayerNorm se adaptan
            #       automaticamente al shape del tensor durante el primer forward pass.
            self.ln_x = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])  # IMAGE_SIZE: normaliza sobre [H_patch, W_patch, num_hidden*7]
            self.ln_h = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])  # IMAGE_SIZE: normaliza sobre [H_patch, W_patch, num_hidden*4]
            self.ln_m = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])  # IMAGE_SIZE: normaliza sobre [H_patch, W_patch, num_hidden*3]
            self.ln_o = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])  # IMAGE_SIZE: normaliza sobre [H_patch, W_patch, num_hidden]

    def call(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)   # [B, H, W, num_hidden*7]
        h_concat = self.conv_h(h_t)   # [B, H, W, num_hidden*4]
        m_concat = self.conv_m(m_t)   # [B, H, W, num_hidden*3]

        if self.use_layer_norm:
            x_concat = self.ln_x(x_concat)
            h_concat = self.ln_h(h_concat)
            m_concat = self.ln_m(m_concat)

        # Dividir por el eje de canales (axis=-1)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = tf.split(
            x_concat, num_or_size_splits=7, axis=-1
        )
        i_h, f_h, g_h, o_h = tf.split(h_concat, num_or_size_splits=4, axis=-1)
        i_m, f_m, g_m      = tf.split(m_concat, num_or_size_splits=3, axis=-1)

        # --- Celda C (memoria espaciotemporal) ---
        i_t = tf.sigmoid(i_x + i_h)
        f_t = tf.sigmoid(f_x + f_h + self._forget_bias)
        g_t = tf.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new   = f_t * c_t + delta_c

        # --- Celda M (memoria temporal zigzag) ---
        i_t_prime = tf.sigmoid(i_x_prime + i_m)
        f_t_prime = tf.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = tf.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new   = f_t_prime * m_t + delta_m

        # --- Puerta de salida combinando C y M ---
        mem = tf.concat([c_new, m_new], axis=-1)   # [B, H, W, num_hidden*2]
        o_mem = self.conv_o(mem)
        if self.use_layer_norm:
            o_mem = self.ln_o(o_mem)
        o_t   = tf.sigmoid(o_x + o_h + o_mem)
        h_new = o_t * tf.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
