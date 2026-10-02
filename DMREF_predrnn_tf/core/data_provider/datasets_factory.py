from DMREF_predrnn_tf.core.data_provider import mnist

# TODO: [PASO 2] Importar el data provider del dataset DMREF y registrarlo aqui.
#       Ejemplo:
#           from DMREF_predrnn_tf.core.data_provider import dmref_dataset
#           datasets_map = { 'mnist': mnist, 'dmref': dmref_dataset }
#       El nuevo data provider debe devolver batches con shape [B, T, H, W, C]
#       en formato channels-last (igual que mnist.py) para que reshape_patch funcione sin cambios.
datasets_map = {
    'mnist': mnist,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths,
                  batch_size, img_width, seq_length, is_training=True):
    """
    Fabrica de data handles.

    Args:
        dataset_name      : str  ('mnist')
        train_data_paths  : str  rutas separadas por coma
        valid_data_paths  : str  rutas separadas por coma
        batch_size        : int
        img_width         : int  (no usado en mnist, incluido por compatibilidad)
        seq_length        : int  total_length
        is_training       : bool

    Returns:
        (train_handle, test_handle)  si is_training=True
        test_handle                  si is_training=False
    """
    if dataset_name not in datasets_map:
        raise ValueError(f'Dataset desconocido: {dataset_name}. '
                         f'Disponibles: {list(datasets_map.keys())}')

    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'mnist':
        test_input_param = {
            'paths':              valid_data_list,
            'minibatch_size':     batch_size,
            'input_data_type':    'float32',
            'is_output_sequence': True,
            'name':               dataset_name + ' test iterator',
        }
        test_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_handle.begin(do_shuffle=False)

        if is_training:
            train_input_param = {
                'paths':              train_data_list,
                'minibatch_size':     batch_size,
                'input_data_type':    'float32',
                'is_output_sequence': True,
                'name':               dataset_name + ' train iterator',
            }
            train_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_handle.begin(do_shuffle=True)
            return train_handle, test_handle
        else:
            return test_handle
