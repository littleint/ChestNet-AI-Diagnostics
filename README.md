---
title: "ChestNet AI: Lung Disease Classifier"
emoji: 🩺
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# 🩺 ChestNet AI: Multi-Class Lung Diagnostics

ChestNet AI is an interactive deep learning prototype designed to classify Chest X-rays into five distinct lung health categories. The model uses a custom **Gated-Attention Feature Pyramid Network (FPN)** backbone with MobileNetV2 and DenseNet121, providing explainable AI (XAI) capabilities through **Grad-CAM attention overlays**.

## 📊 Classification Categories
The model predicts five conditions:
*   **Bacterial Pneumonia**
*   **Corona Virus Disease (COVID-19)**
*   **Normal**
*   **Tuberculosis**
*   **Viral Pneumonia**

---

## 📂 Project Directory Structure

```
.
├── .gitattributes          # Git LFS configuration for heavy model files
├── README.md               # GitHub docs & Hugging Face Space configuration
├── app.py                  # Main Gradio application entry point
├── best_model.keras        # Trained Keras model file (~113 MB)
├── class_mapping.json      # Mapping from class indices to labels
├── requirements.txt        # Project python package requirements
├── notebooks/              # Directory for model notebooks
│   └── lung-disease-detection.ipynb
└── samples/                # Directory containing reference images for testing
    ├── normal.jpg          # Reference normal radiograph
    └── pneumonia.jpg       # Reference pneumonia radiograph
```

---

## 💻 Local Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd Xray
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

## 🌐 Deploying to Hugging Face Spaces

Hugging Face Spaces allows you to host this Gradio app for free. Follow these steps to deploy:

1. **Create a Space**:
   * Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   * Enter a space name.
   * Select **Gradio** as the SDK.
   * Choose the **CPU Basic** hardware (free tier is sufficient).

2. **Upload Repository Files**:
   Since the model file `best_model.keras` exceeds 100MB, you must configure **Git LFS** when pushing your repository:
   ```bash
   # Initialize Git
   git init
   
   # Install Git LFS (if not already installed)
   git lfs install

   # Track Keras files
   git lfs track "*.keras"

   # Add files and commit
   git add .
   git commit -m "Initial commit with FPN Gated-Attention model and Gradio app"

   # Add Hugging Face Space as a remote and push
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push -f hf master
   ```

Hugging Face will automatically install the packages listed in `requirements.txt`, read the configuration in `README.md`, load the model, and deploy the Gradio app!
