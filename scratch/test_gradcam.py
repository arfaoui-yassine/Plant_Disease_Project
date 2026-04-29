import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("outputs/notebook_run/dl/dl_model.keras")

inner_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        inner_model = layer
        break

x = np.zeros((1, 224, 224, 3), dtype=np.float32)
try:
    for layer in model.layers:
        if layer == inner_model:
            break
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        print(f"Calling layer: {layer}")
        x = layer(x)
    print("Preprocess passed.")
    
    inner_grad_model = tf.keras.models.Model(
        inner_model.inputs, 
        [inner_model.get_layer("out_relu").output, inner_model.output]
    )
    last_conv_layer_output, base_outputs = inner_grad_model(x)
    print("Inner grad model passed.")
    
    past_inner = False
    for layer in model.layers:
        if past_inner:
            print(f"Calling layer post-inner: {layer}")
            base_outputs = layer(base_outputs)
        if layer == inner_model:
            past_inner = True
    print("Success")
except Exception as e:
    print(f"Error: {e}")
