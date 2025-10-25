# 🎯 Emotion Classification Using Facial Expressions and EEG  

This repository contains all code, data pipelines, models, and notebooks developed as part of the **IITB EdTech Internship 2025 (Group T1_G38 – Team_Attackers)** for **Problem ID 4: Emotion Classification using Facial Expressions and EEG**.

---

## 📑 Project Overview  

**Objective:** Predict emotions (e.g. Engaged, Confused, Neutral) from EEG and Affectiva facial expression data.  

**Modalities:**  
- **EEG**: Delta, Theta, Alpha, Beta, Gamma bands  
- **Facial Expressions**: Affectiva probabilities & Action Units from TIVA  

**Key Challenges:** Temporal alignment of multimodal data and class imbalance.  

---

## 📂 Project Structure  

Based on the project structure you've provided, here is a correct and complete description for your `README.md` file. It consolidates all the information from your screenshots and the project plan into a clear, single format.

-----

### 📂 Project Structure

```
project/
├── data/
│   ├── EEG_combined.csv # Preprocessed and synchronized EEG features.
│   ├── PSY_combined.csv # Preprocessed and synchronized behavioral data.
│   ├── TIVA_combined.csv # Preprocessed and synchronized Affectiva facial features.
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
│   ├── rf_allfeatures.pkl # A trained Scikit-learn Random Forest model on all features.
│   ├── xg_allfeatures.pkl # A trained Scikit-learn XGBoost model on all features.
│   ├── logreg_scalefeatures.pkl # A trained Scikit-learn Logistic Regression model on scaled features.
│   ├── rf_eeg.pkl # A trained Random Forest model using only EEG features.
│   ├── rf_tiva_latefusion.pkl # A trained Random Forest model using only TIVA features for late fusion.
│   ├── rf_eeg_latefusion.pkl # A trained Random Forest model using only EEG features for late fusion.
│   ├── rf_earlyfusion.pkl # A trained Random Forest model on concatenated EEG and TIVA features.
│   ├── mlp_intermidatefusion.pkl # A trained MLP model for intermediate fusion.
│   └── multimodal_bilstm_attention.pth # A trained PyTorch BiLSTM model with attention.
└── README.md # The project's main documentation and overview.
```





## 🗂 Data Access
Raw and processed data are hosted on Google Drive: 📥 IITB Dataset Folder

Within Drive:
[📥 IITB Dataset Folder](https://drive.google.com/drive/folders/1fyx2rXwoAhC3iiwoq6tDTyE0XjnjhMYx?usp=sharing)

IITB/Data/processed contains combined/cleaned data used in this repository.

IITB/project/ contains notebooks and scripts.


## 🗂 Data Access  

Extracted and processed data are hosted on Google Drive:  
[📥 project Dataset Folder](https://drive.google.com/drive/folders/1t0SB-wcesioeYdzdGwnte3m-vxgmYkCc?usp=sharing)


**How to Use the Data:**  
1. Download from the Google Drive link above.  
2. In your local repository, create a `data/` folder (if not already).  
3. Move the downloaded files (`EEG_combined.csv`, `TIVA_combined.csv`, `PSY_combined.csv`, `WINDOW_combined.csv`, `df_features_trials.csv`) into the `data/` folder.  
4. Run the notebooks – they will automatically load the data from `data/`.  

---

## 🚀 Steps Implemented   

### 1️⃣ Preprocessing  
- Combined EEG, TIVA, PSY data across 38 students  
- Cleaned missing values & handled NaNs  
- Synchronized timestamps using `routineStart`/`routineEnd`  
- Downsampled to ~2.3 Hz per label  

### 2️⃣ Feature Engineering  
- **EEG**: mean/variance for Delta, Theta, Alpha, Beta, Gamma; frontal asymmetry, engagement index  
- **Facial**: raw Affectiva emotion probabilities (Joy, Anger, Fear…) and Action Units (AUs) aggregated per trial (mean/std/max/occurrence)  

### 3️⃣ Modeling  
- **Baseline (03_modeling_baseline):** RF, XGBoost, Logistic Regression on concatenated EEG+TIVA features. Models:  
  - `rf_allfeatures.pkl`  
  - `xg_allfeatures.pkl`  
  - `logreg_scalefeatures.pkl`  

- **Fusion (05_modeling_fusion):**  
  - **Early fusion:** concatenate EEG+facial features → classifier (`rf_earlyfusion.pkl`)  
  - **Late fusion:** separate models per modality, combine logits (`rf_eeg_latefusion.pkl`, `rf_tiva_latefusion.pkl`)  
  - **Intermediate fusion:** embeddings from CNN + EEG MLP → joint classifier (`mlp_intermidatefusion.pkl`)  

- **Advanced (04_temporal_models):** BiLSTM with attention for temporal modeling (`multimodal_bilstm_attention.pth`)  

### 4️⃣ Evaluation & Interpretation (06_analysis)  
- Accuracy, macro F1, precision, recall  
- Confusion Matrix per class  
- ROC-AUC for binary subsets (e.g., Engaged vs Not Engaged)  
- SHAP values to interpret EEG bandpower and AUs importance  
- Attention weights visualization to see modality dominance  

### 5️⃣ Experimentation & Improvement  
- Tested different time-window sizes  
- Oversampling with SMOTE for underrepresented emotions  
- Participant-specific vs cross-participant splits  
- Continuous dimensions (Valence, Arousal)  
- Temporal smoothing of predictions (HMM / majority vote)  

---

## 📊 How to Reproduce  

1. Clone this repo.  
2. Install requirements:
   ```bash
   pip install -r requirements.txt
3. Mount your Google Drive or ensure data/ and models/ are in place.

4. Run notebooks in order:
  → 01_preprocessing.ipynb 
  → 02_feature_engineering.ipynb 
  → 03_modeling_baseline.ipynb 
  → 04_temporal_models.ipynb 
  → 05_modeling_fusion.ipynb 
  → 06_analysis.ipynb

5. Load trained models:
  import joblib
  rf = joblib.load('models/rf_allfeatures.pkl')

  import torch
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

## 📊 Detailed Results per Class 

Below are the **per-class evaluation results** (precision, recall, F1) for each model.  

### 🔹 Random Forest (All Features) – `rf_allfeatures.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.73      | 0.78   | 0.76     | 208     |
| INCORRECT | 0.27      | 0.22   | 0.24     | 67      |
| SKIP      | 0.10      | 0.14   | 0.12     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.61** | 290     |

---

### 🔹 XGBoost (All Features) – `xg_allfeatures.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.74      | 0.81   | 0.78     | 208     |
| INCORRECT | 0.33      | 0.25   | 0.29     | 67      |
| SKIP      | 0.00      | 0.00   | 0.00     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.64** | 290     |

---

### 🔹 Logistic Regression (Scaled Features) – `logreg_scalefeatures.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.69      | 0.47   | 0.56     | 208     |
| INCORRECT | 0.22      | 0.24   | 0.23     | 67      |
| SKIP      | 0.10      | 0.43   | 0.16     | 7       |
| Unknown   | 0.02      | 0.12   | 0.04     | 8       |
| **Accuracy** |       |        | **0.41** | 290     |

---

### 🔹 Random Forest (EEG Only) – `rf_eeg.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.72      | 1.00   | 0.83     | 208     |
| INCORRECT | 0.00      | 0.00   | 0.00     | 67      |
| SKIP      | 0.00      | 0.00   | 0.00     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.71** | 290     |

---

### 🔹 Random Forest (Early Fusion) – `rf_earlyfusion.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.71      | 0.77   | 0.74     | 208     |
| INCORRECT | 0.25      | 0.21   | 0.23     | 67      |
| SKIP      | 0.20      | 0.14   | 0.17     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.60** | 290     |

---

### 🔹 Random Forest (Late Fusion – EEG+TIVA) – `rf_eeg_latefusion.pkl` + `rf_tiva_latefusion.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.72      | 1.00   | 0.84     | 208     |
| INCORRECT | 0.00      | 0.00   | 0.00     | 67      |
| SKIP      | 0.00      | 0.00   | 0.00     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.72** | 290     |

---

### 🔹 MLP (Intermediate Fusion) – `mlp_intermidatefusion.pkl`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.72      | 0.80   | 0.76     | 208     |
| INCORRECT | 0.21      | 0.13   | 0.17     | 67      |
| SKIP      | 0.11      | 0.14   | 0.12     | 7       |
| Unknown   | 0.00      | 0.00   | 0.00     | 8       |
| **Accuracy** |       |        | **0.61** | 290     |

---

### 🔹 BiLSTM + Attention – `multimodal_bilstm_attention.pth`
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CORRECT   | 0.73      | 0.78   | 0.76     | 134     |
| INCORRECT | 0.26      | 0.22   | 0.24     | 50      |
| SKIP      | 0.00      | 0.00   | 0.00     | 5       |
| Unknown   | 0.00      | 0.00   | 0.00     | 4       |
| **Accuracy** |       |        | **0.60** | 193     |


📝 Notes

All .pkl models use joblib.

PyTorch models are saved as .pth.


👥 Team

Group T1_G38 – Team_Attackers

Members:

Pranav Chandrakant Patil

Harshad Balaso Kanire

Omkar Manik Patil

Faculty Mentor: Mr. Swapnil T. Powar
 

7. 
