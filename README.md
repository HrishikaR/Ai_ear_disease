# Ai_ear_disease
Multimodal Explainable AI for Ear Disease Classification

An AI-based system for ear disease classification that combines otoscopic image analysis and patient-reported symptoms using a multimodal deep learning approach. The system uses a Vision Transformer (ViT) for image-based classification, a Random Forest classifier for symptom-based prediction, and weighted late fusion to combine the outputs. It also incorporates Grad-CAM, SHAP, and Gemini API to provide visual and textual explanations.

Project Overview

Early identification of ear diseases can help support timely medical evaluation. This project explores a multimodal approach that analyzes both otoscopic images and reported symptoms to classify ear conditions.

The system combines two modalities:

- Image-based classification: A pretrained Vision Transformer (ViT) analyzes otoscopic images and predicts the corresponding ear condition.
- Symptom-based classification: A Random Forest classifier analyzes patient-reported symptoms represented as binary features.
- Weighted late fusion: The probability outputs from both models are combined to generate the final prediction.
- Explainable AI: Grad-CAM highlights influential image regions, while SHAP identifies symptom features contributing to the prediction.
- AI-generated explanation: Gemini API generates a natural-language explanation using the model prediction and explainability outputs.

Diseases Covered

The image classification model supports the following five categories:

1. Chronic Otitis Media
2. Cerumen Impaction (Earwax)
3. Acute Otitis Media
4. Myringosclerosis
5. Normal

The symptom classification model uses four categories: Chronic Otitis Media, Cerumen Impaction, Myringosclerosis, and Normal.

Technologies Used

Category| Technologies
Programming Language| Python
Deep Learning| PyTorch
Image Classification| Vision Transformer (ViT)
Machine Learning| Scikit-learn, Random Forest
Explainable AI| Grad-CAM, SHAP
Generative AI| Gemini API
Interface| Gradio
Data Processing| NumPy, Pandas
Image Processing| PIL, OpenCV
Visualization| Matplotlib, Seaborn

System Architecture

The proposed system follows a multimodal classification pipeline:

1. Input: An otoscopic image and patient-reported symptoms are provided.

2. Image Processing: The image is resized to 224 × 224 pixels, normalized, and passed to the pretrained Vision Transformer.

3. Image Prediction: ViT generates class probabilities for the five image categories.

4. Symptom Processing: The reported symptoms are converted into binary features and passed to the Random Forest classifier.

5. Probability Alignment: Symptom-model probabilities are aligned with the image model's five-class output space.

6. Weighted Late Fusion: Both probability distributions are combined using the following weights:
   
   "Final Probability = (0.8 × Image Probability) + (0.2 × Symptom Probability)"

7. Explainability: Grad-CAM and SHAP provide insights into the image and symptom-based predictions.

8. Explanation Generation: Gemini API generates a textual explanation based on the prediction and explainability information.

Symptom Features

The Random Forest model uses nine binary symptom features:

- Pain
- Discharge
- Hearing Loss
- Fever
- Blocked Ear
- Itching
- Dizziness
- Tinnitus
- Long Duration

The symptom dataset consists of 500 synthetic records generated with the assistance of AI tools based on predefined disease categories and symptom information.

Explainable AI

Grad-CAM

Grad-CAM generates visual heatmaps highlighting image regions that influence the Vision Transformer’s predictions.

SHAP

SHAP explains the contribution of individual symptom features to the Random Forest model's predictions.

Gemini-Powered Explanation

The Gemini API generates a natural-language explanation incorporating the predicted condition, relevant symptom contributions, and image-analysis information.

Application Features

- User registration and login
- Multimodal ear disease prediction
- Otoscopic image upload
- Symptom-based input
- Weighted probability fusion
- Grad-CAM visualizations
- SHAP-based symptom explanations
- Gemini-powered textual explanations
- Prediction history
- Admin dashboard

Dataset

Otoscopic Image Dataset: "UCI Otoscopic Image Dataset – Kaggle" (https://www.kaggle.com/datasets/ucimachinelearning/otoscopic-image-dataset)

The image dataset is downloaded using "kagglehub".

The symptom dataset is a synthetic dataset containing 500 records and nine binary symptom features.

Installation and Execution

1. Clone the Repository

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Install Dependencies

pip install torch torchvision timm
pip install scikit-learn pandas numpy
pip install matplotlib seaborn pillow opencv-python
pip install grad-cam shap
pip install gradio
pip install kagglehub google-generativeai joblib

3. Configure API Key

Configure your Gemini API key securely using Google Colab Secrets or an environment variable. Do not hardcode or upload API keys to GitHub.

4. Prepare the Dataset

- Download the otoscopic image dataset using KaggleHub.
- Place the symptom dataset file "symptom_dataset_500_rows_v2.csv" in the notebook's working directory.
- Ensure the image dataset is organized into class-wise folders.

5. Run the Project

Open the Jupyter Notebook in Google Colab or a compatible Python environment and execute the cells in sequence.

The notebook trains the models, performs multimodal fusion, generates explainability outputs, and launches the Gradio application.

Model Files

The trained models are saved as:

- "vit_ear_model.pth" – Vision Transformer model weights.
- "symptom_rf_model.pkl" – Random Forest symptom classifier.

These files are required for loading the trained models during inference.

Important Notes and Limitations

- The symptom dataset is synthetic and does not represent actual patient records.
- The image and symptom datasets are separate and are not linked through patient-level records.
- The fusion weights are predefined for the current prototype.
- Model predictions and AI-generated explanations are experimental and require validation using real, paired clinical data.
- Grad-CAM and SHAP provide model explanations, not proof of clinical reasoning.
- This project is intended for academic and research purposes and is not a substitute for professional medical diagnosis.

Future Enhancements

- Incorporating larger, real-world paired image and symptom datasets.
- Optimizing fusion weights and exploring advanced fusion strategies.
- Improving model generalization through extensive training and external validation.
- Enhancing prediction reliability and explainability.
- Strengthening user authentication and data security.

Disclaimer

This project is developed for academic and research purposes only. It is an experimental AI-based screening and classification system and should not be used as a standalone diagnostic tool or for making medical decisions.
