import os
import shutil
import json
import numpy as np
import tensorflow as tf
import keras
import gradio as gr
from PIL import Image
import cv2

# ---------- Step 1: Set up sample directories ----------
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

dst_normal = os.path.join(SAMPLES_DIR, "normal.jpg")
dst_pneumonia = os.path.join(SAMPLES_DIR, "pneumonia.jpg")

# ---------- Step 2: Define Custom ResizeToLike Layer ----------
@keras.saving.register_keras_serializable(name='ResizeToLike')
class ResizeToLike(keras.layers.Layer):
    def __init__(self, method='nearest', **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs, **kwargs):
        x, ref = inputs
        ref_shape = tf.shape(ref)[1:3]
        return tf.image.resize(x, ref_shape, method=self.method)

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method})
        return config

# ---------- Step 3: Load Model & Class Mapping ----------
MODEL_PATH = "best_model.keras"
CLASS_MAPPING_PATH = "class_mapping.json"

print("Loading Keras model...")
model = keras.models.load_model(MODEL_PATH, custom_objects={"ResizeToLike": ResizeToLike})
print("Model loaded successfully!")

with open(CLASS_MAPPING_PATH, "r") as f:
    class_mapping = json.load(f)

# ---------- Step 4: Grad-CAM Implementation ----------
def generate_gradcam(img_array, model, last_conv_layer_name='merged_pyramid', pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # Feed inputs via dictionary matching the model input name to prevent warnings/errors
        last_conv_layer_output, preds = grad_model({"input_image": img_array})
        
        # Check if preds is nested inside a list
        if isinstance(preds, list):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    img = np.array(img_pil.convert("RGB"))
    h, w, _ = img.shape
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_scaled = np.uint8(255 * heatmap_resized)
    
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_colored * alpha + img * (1.0 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

# ---------- Step 5: Prediction Pipeline ----------
def predict_xray(img_pil):
    if img_pil is None:
        return None, None
    
    orig_img = img_pil.convert("RGB")
    img_resized = orig_img.resize((224, 224))
    
    # Rescale pixels to [0, 1] range as done during model training
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Feed input through dictionary
    preds = model.predict({"input_image": img_array}, verbose=0)
    
    if isinstance(preds, list):
        preds = preds[0]
    preds = preds[0]
    
    confidences = {
        class_mapping.get(str(i), f"Class {i}"): float(preds[i])
        for i in range(len(preds))
    }
    
    top_index = int(np.argmax(preds))
    
    # Generate heatmap for the top class
    heatmap = generate_gradcam(img_array, model, pred_index=top_index)
    gradcam_img = overlay_heatmap(orig_img, heatmap, alpha=0.45)
    
    return confidences, gradcam_img

# ---------- Step 6: Simplified Gradio Interface ----------
with gr.Blocks(title="ChestNet AI") as demo:
    gr.Markdown("# 🩺 ChestNet AI: Lung Disease Classifier")
    gr.Markdown("Upload a chest X-ray image to classify lung disease and visualize Grad-CAM model attention.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload X-ray Image")
            btn_predict = gr.Button("Analyze Image", variant="primary")
            
            gr.Examples(
                examples=[
                    os.path.join(SAMPLES_DIR, "normal.jpg"),
                    os.path.join(SAMPLES_DIR, "pneumonia.jpg")
                ],
                inputs=input_image,
                label="Example Radiographs"
            )
            
        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Diagnosis Probability")
            output_gradcam = gr.Image(label="Grad-CAM Pathology Attention Overlay")
            
    btn_predict.click(
        fn=predict_xray,
        inputs=input_image,
        outputs=[output_label, output_gradcam]
    )

if __name__ == "__main__":
    # Bind to PORT environment variable injected by hosting providers like Render
    port = int(os.environ.get("PORT", 7860))
    # Bind to 0.0.0.0 to accept external traffic in containerized environments
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
