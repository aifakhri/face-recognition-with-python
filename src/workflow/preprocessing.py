import numpy as np

import tensorflow as tf



def image_encoding(img_path, model):
    """Function to encode the image
    """

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)

    x_train = np.expand_dims(img, axis=0)

    embed = model.predict_on_batch(x_train)

    return embed / np.linalg.norm(embed, ord=2)