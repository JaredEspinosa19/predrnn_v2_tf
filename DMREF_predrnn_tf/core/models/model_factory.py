import os
import tensorflow as tf
from DMREF_predrnn_tf.core.models import predrnn_v2


class Model:
    """
    Wrapper de alto nivel que encapsula el modelo, el optimizador y
    las operaciones de entrenamiento/test.

    Equivalente a core/models/model_factory.py de la version PyTorch,
    pero usando tf.GradientTape en lugar de loss.backward().

    Uso basico:
        model = Model(configs)
        loss  = model.train(frames_np, mask_np)
        preds = model.test(frames_np, mask_np)
        model.save(itr)
        model.load('checkpoints/model-1000')
    """

    def __init__(self, configs):
        self.configs    = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)

        networks_map = {
            'predrnn_v2': predrnn_v2.RNN,
        }

        if configs.model_name not in networks_map:
            raise ValueError(f'Modelo desconocido: {configs.model_name}. '
                             f'Disponibles: {list(networks_map.keys())}')

        Network       = networks_map[configs.model_name]
        self.network  = Network(self.num_layers, self.num_hidden, configs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=configs.lr)

        # Compilar la funcion de entrenamiento una vez para eficiencia
        self._train_step = tf.function(self._train_step_fn)

    def _train_step_fn(self, frames, mask):
        with tf.GradientTape() as tape:
            _, loss = self.network(frames, mask, training=True)
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return loss

    def train(self, frames, mask):
        """
        Un paso de entrenamiento.

        Args:
            frames : np.array  [B, T, H, W, C]  (en espacio de patches)
            mask   : np.array  [B, T-1, H, W, C]

        Returns:
            loss_value : float escalar
        """
        frames_tensor = tf.cast(frames, tf.float32)
        mask_tensor   = tf.cast(mask,   tf.float32)
        loss = self._train_step(frames_tensor, mask_tensor)
        return float(loss.numpy())

    def test(self, frames, mask):
        """
        Inferencia sin calculo de gradientes.

        Args:
            frames : np.array  [B, T, H, W, C]
            mask   : np.array  [B, T-1, H, W, C]  (normalmente ceros en test)

        Returns:
            next_frames : np.array  [B, T-1, H, W, C]
        """
        frames_tensor = tf.cast(frames, tf.float32)
        mask_tensor   = tf.cast(mask,   tf.float32)
        next_frames, _ = self.network(frames_tensor, mask_tensor, training=False)
        return next_frames.numpy()

    def save(self, itr):
        """Guarda los pesos del modelo en formato TF checkpoint."""
        os.makedirs(self.configs.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.configs.save_dir, f'model-{itr}')
        self.network.save_weights(checkpoint_path)
        print(f'Modelo guardado en {checkpoint_path}')

    def load(self, checkpoint_path):
        """
        Carga pesos desde un checkpoint.
        Se requiere haber ejecutado al menos un forward pass antes
        para que las variables esten inicializadas.
        """
        print(f'Cargando modelo desde {checkpoint_path}')
        self.network.load_weights(checkpoint_path)
