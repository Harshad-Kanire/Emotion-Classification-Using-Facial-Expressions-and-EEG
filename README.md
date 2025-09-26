Emotion Classification Using Facial Expressions and EEG

This repository contains all code, data pipelines, models, and notebooks developed as part of the IITB EdTech Internship 2025 (Group T1_G38 – Team_Attackers) for Problem ID 4: Emotion Classification using Facial Expressions and EEG.
📑 Project Overview

Objective: Predict emotions (e.g. Engaged, Confused, Neutral) from EEG and Affectiva facial expression data.

Modalities:

EEG (Delta, Theta, Alpha, Beta, Gamma bands)

Facial Expressions (Affectiva probabilities & Action Units from TIVA)

Key Challenges: Temporal alignment of multimodal data and class imbalance.

project/
├── data/
│   ├── EEG_combined.csv # Contains preprocessed and synchronized EEG features.
│   ├── PSY_combined.csv # Contains preprocessed and synchronized TIVA behavioral data.
│   ├── TIVA_combined.csv # Contains preprocessed and synchronized Affectiva facial features.
│   ├── WINDOW_combined.csv # The core dataset after applying the sliding window approach.
│   └── df_features_trials.csv # A trial-level dataset with final labels and features.
├── notebooks/
│   ├── 01_preprocessing.ipynb # Notebook for data cleaning, merging, and initial synchronization.
│   ├── 02_feature_engineering.ipynb # Notebook for extracting and combining multimodal features.
│   ├── 03_modeling_baseline.ipynb # Notebook for training and evaluating baseline ML models (RF, XGBoost, LR).
│   ├── 04_temporal_models.ipynb # Notebook dedicated to implementing and training the BiLSTM with attention.
│   ├── 05_modeling_fusion.ipynb # Notebook for exploring different fusion techniques (early, late, intermediate).
│   └── 06_analysis.ipynb # Notebook for in-depth model evaluation, interpretation, and visualization.
├── models/
│   ├── rf_allfeatures.pkl # A trained Scikit-learn Random Forest model.
│   ├── xg_allfeatures.pkl # A trained Scikit-learn XGBoost model.
│   ├── logreg_scalefeatures.pkl # A trained Scikit-learn Logistic Regression model.
│   ├── rf_eeg.pkl # A trained Random Forest model using only EEG features.
│   ├── rf_tiva_latefusion.pkl # A trained Random Forest model using only TIVA features for late fusion.
│   ├── rf_eeg_latefusion.pkl # A trained Random Forest model using only EEG features for late fusion.
│   ├── rf_earlyfusion.pkl # A trained Random Forest model on concatenated EEG and TIVA features.
│   ├── mlp_intermidatefusion.pkl # A trained MLP model for intermediate fusion.
│   └── multimodal_bilstm_attention.pth # A trained PyTorch BiLSTM model with attention.
└── README.md # The project's main documentation and overview.

🗂 Data Access
Raw and processed data are hosted on Google Drive: 📥 IITB Dataset Folder

Within Drive:

IITB/Data/processed contains combined/cleaned data used in this repository.

IITB/project/ contains notebooks and scripts.

🚀 Steps Implemented (matching internship tasks)
1️⃣ Preprocessing
Combined EEG, TIVA, PSY data across 38 students.

Cleaned missing values & handled NaNs.

Synchronized timestamps using routineStart/routineEnd.

Downsampled to ~2.3 Hz per label.

2️⃣ Feature Engineering
EEG: mean/variance for Delta, Theta, Alpha, Beta, Gamma; frontal asymmetry, engagement index.

Facial: raw Affectiva emotion probabilities (Joy, Anger, Fear…) and Action Units (AUs) aggregated per trial (mean/std/max/occurrence).

3️⃣ Modeling
Baseline (03_modeling_baseline): RF, XGBoost, Logistic Regression on concatenated EEG+TIVA features. The saved models include rf_allfeatures.pkl, xg_allfeatures.pkl, and logreg_scalefeatures.pkl.

Fusion (05_modeling_fusion):

Early fusion: concatenate EEG+facial features → classifier. The model rf_earlyfusion.pkl is an example of this approach.

Late fusion: separate models per modality, combine logits. The models rf_eeg_latefusion.pkl and rf_tiva_latefusion.pkl represent this approach.

Intermediate fusion: embeddings from CNN + EEG MLP → joint classifier. The saved model is mlp_intermidatefusion.pkl.

Advanced: BiLSTM with attention for temporal modeling, saved as multimodal_bilstm_attention.pth. This is implemented in the 04_temporal_models.ipynb notebook.

4️⃣ Evaluation & Interpretation (06_analysis)
Accuracy, macro F1, precision, recall.

Confusion Matrix per class.

ROC-AUC for binary subsets (e.g., Engaged vs Not Engaged).

SHAP values to interpret EEG bandpower and AUs importance.

Attention weights visualization to see modality dominance.

5️⃣ Experimentation & Improvement
Tested different time-window sizes.

Oversampling with SMOTE for underrepresented emotions.

Participant-specific vs cross-participant splits.

Continuous dimensions (Valence, Arousal).

Temporal smoothing of predictions (HMM / majority vote).

📊 How to Reproduce
Clone this repo.

Install requirements:

Bash

pip install -r requirements.txt
This will install pandas, numpy, scikit-learn, torch, xgboost, and other necessary libraries.

Mount your Google Drive or ensure data/ and models/ are in place.

Run notebooks in order:

01_preprocessing.ipynb -> 02_feature_engineering.ipynb -> 03_modeling_baseline.ipynb -> 04_temporal_models.ipynb -> 05_modeling_fusion.ipynb -> 06_analysis.ipynb

Models are saved automatically in models/ after training:
import joblib
rf = joblib.load('models/rf_allfeatures.pkl')

For PyTorch models:
import torch
# Replace MyBiLSTMModel with your actual model class
model = MyBiLSTMModel(...)
model.load_state_dict(torch.load('models/multimodal_bilstm_attention.pth'))
model.eval()

## 📈 Results (Test Set)

| Model                          | Accuracy | Macro F1 |
|-------------------------------|----------|----------|
| Random Forest (All Features)  | 0.61     | 0.28     |
| XGBoost (All Features)        | 0.64     | 0.27     |
| Logistic Regression (Scaled)  | 0.41     | 0.25     |
| Random Forest (EEG only)      | 0.71     | 0.21     |
| Random Forest (Early Fusion)  | 0.60     | 0.28     |
| Random Forest (Late Fusion)   | 0.72     | 0.21     |
| MLP (Intermediate Fusion)     | 0.61     | 0.26     |
| BiLSTM + Attention            | 0.60     | 0.25     |

📝 Notes

All .pkl models use joblib .

PyTorch models are saved as .pth.


👥 Team

Group T1_G38 – Team_Attackers

Group Members:1)Pranav Chandrakant Patil
              2)Harshad Balaso Kanire
              3)Omkar Manik patil
Faculty Mentor: Mr. Swapnil T. Powar
