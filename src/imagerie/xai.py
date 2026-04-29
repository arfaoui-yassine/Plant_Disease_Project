import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image and model, specifically handling nested base models.
    """
    # 1. Find the nested base model (MobileNetV2)
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
            
    if inner_model is None:
        raise ValueError("Could not find nested inner model.")

    # 2. The true last conv layer is inside the inner_model. 
    # For MobileNetV2, it's typically "out_relu" or "Conv_1"
    # We will look for "out_relu" explicitly if the passed name was the base model name.
    if last_conv_layer_name in [inner_model.name, "mobilenet_v2_base"]:
        last_conv_layer_name = "out_relu"

    # 3. Create a model that maps the inner model's input to its last conv layer and its output
    inner_grad_model = tf.keras.models.Model(
        [inner_model.inputs], 
        [inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
    )

    with tf.GradientTape() as tape:
        # Pass input through any preprocessing layers before the inner model
        x = img_array
        for layer in model.layers:
            if layer == inner_model:
                break
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            
        # Get conv features and base model output
        last_conv_layer_output, base_outputs = inner_grad_model(x)
        
        # Pass base output through the classifier layers (pooling, dense, etc.)
        x = base_outputs
        past_inner = False
        for layer in model.layers:
            if past_inner:
                x = layer(x)
            if layer == inner_model:
                past_inner = True
                
        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of the predicted class with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map by the pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    """
    Superimpose Grad-CAM heatmap on the original image.
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img / 255.0
    superimposed_img = np.clip(superimposed_img, 0, 1)
    superimposed_img = np.uint8(255 * superimposed_img)

    return superimposed_img
