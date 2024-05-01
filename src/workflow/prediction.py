import numpy as np
import tensorflow as tf

from preprocessing import image_encoding



def predict_image(img_fl, database, model):
    """
    """

    encoding_ = image_encoding(img_fl, model)

    min_dist = 100

    for (_, db_enc) in database.items():
        distance = np.linalg.norm(tf.substract(db_enc, encoding_))

        if distance < min_dist:
            min_dist = distance

    if min_dist > 0.7:
        state = False
    else:
        state = True

    return state