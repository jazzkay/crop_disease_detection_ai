import tensorflow as tf
import numpy as np
from PIL import Image
from utils.class_names import CLASS_NAMES

model = tf.keras.models.load_model("model/trained_model.h5", compile=False)


def predict_pil_image(img):
    img_resized = img.resize((224,224))
    arr = np.array(img_resized)/255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top3 = preds.argsort()[-3:][::-1]

    return [(CLASS_NAMES[i], float(preds[i])) for i in top3]
