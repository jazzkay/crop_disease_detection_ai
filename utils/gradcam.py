import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = tf.pow(heatmap, 1.5)

    return heatmap.numpy()

def generate_gradcam(model, pil_image, last_conv_layer="block_16_project"):

    img = pil_image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    heatmap = make_gradcam_heatmap(model, img, last_conv_layer)

    heatmap = cv2.resize(heatmap, (224,224))

    # Keep only strong activations
    heatmap[heatmap < 0.2] = 0

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    original = np.array(pil_image.resize((224,224)))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)



    return overlay, heatmap

