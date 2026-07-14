# 🩺 ChestNet AI: Multi-Class Lung Diagnostics

ChestNet AI is an interactive deep learning prototype designed to classify Chest X-rays into five distinct lung health categories. The model leverages a custom **Gated-Attention Feature Pyramid Network (FPN)** backbone that fuses features from MobileNetV2 and DenseNet121. To assist with clinical interpretability, the application generates **Grad-CAM attention overlays** to highlight the spatial regions influencing the model's classifications.

---

## 📊 Diagnostic Categories
The model predicts the following five lung conditions:
*   **Bacterial Pneumonia**: Characterized by localized lobar or segmental airspace consolidation.
*   **Corona Virus Disease (COVID-19)**: Typically presents as bilateral, peripheral ground-glass opacities (GGOs) or consolidations.
*   **Normal**: Clear lung fields with no significant consolidations, effusions, or abnormal masses.
*   **Tuberculosis**: Often characterized by apical infiltrates, cavitary lung lesions, or hilar lymphadenopathy.
*   **Viral Pneumonia**: Presents as diffuse, non-lobar bilateral interstitial infiltrates or patchy opacities.

---

## 📂 Project Directory Structure

```
.
├── .gitattributes          # Git LFS configuration for binary model and image files
├── .gitignore              # Ignores python caches, notebook checkpoints, and IDE folders
├── README.md               # Project documentation and deployment guide
├── app.py                  # Main Gradio application containing model pipeline and UI
├── best_model.keras        # Trained FPN Gated-Attention Keras model file (~113 MB)
├── class_mapping.json      # JSON mapping class indices to pathology labels
├── requirements.txt        # Python package dependencies
├── notebooks/              # Directory for model notebooks
│   └── lung-disease-detection.ipynb
└── samples/                # Directory containing test images for quick prototyping
    ├── normal.jpg          # Sample normal chest X-ray
    └── pneumonia.jpg       # Sample chest X-ray with bacterial pneumonia
```

---

## 💻 Local Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/littleint/ChestNet-AI-Diagnostics.git
   cd ChestNet-AI-Diagnostics
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:7860`.

---

## ☁️ Deploying to Render

Render is a cloud hosting platform where you can deploy this Gradio app as a Web Service.

### ⚙️ Deployment Instructions

1. **Create Web Service**:
   * Log in to your account at [dashboard.render.com](https://dashboard.render.com).
   * Click **New +** and select **Web Service**.
   * Connect your GitHub account and select the **`ChestNet-AI-Diagnostics`** repository.

2. **Configure Environment Settings**:
   * **Language**: `Python`
   * **Branch**: `main`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `python -u app.py`

3. **Instance Type (Important)**:
   * **Do not use the Free tier (512 MB RAM)**. Loading TensorFlow CPU and compiles the custom `ResizeToLike` layer requires at least **1.5 GB to 2 GB of RAM** during boot. The free tier will crash with an Out-of-Memory (OOM) error (Exit Code `137`).
   * Choose the **Starter tier (2 GB RAM)** or higher.

Render will automatically inject a `PORT` environment variable. The Gradio app is pre-configured to bind dynamically to this port and listen on host `0.0.0.0` for web routing.

---

## 🩺 Explainable AI: Grad-CAM
The application computes gradients of the top predicted class score with respect to the `merged_pyramid` FPN layer (shape: `56x56x640`). These gradients are global-average-pooled and used to weight the activation maps. The resulting heatmap is resized and superimposed back onto the original radiograph, visually highlighting the pathological focus regions.

---

## ⚠️ Disclaimer
*This AI application is a prototype built for prototyping, educational, and research purposes. It is not a certified medical device and is not intended for clinical diagnostics or decision-making.*
