import os
import yaml
import tensorflow as tf


from tensorflow.python.keras.models import load_model


CONFIG_PATH = "config/config.yaml"



def load_config():
    """Load configuration file
    """

    with open(CONFIG_PATH, "r") as fl:
        config = yaml.safe_load(fl)

    return config


def load_facenet(model_path):
    """Function to load pre-trained FaceNet model
    """

    model = load_model(model_path)

    return model

def load_image(image_path):
    """Function to load image
    """

    img = tf.keras.preprocessing.image.load_img(image_path)

    return img

def image_database():
    """
    """

    pass