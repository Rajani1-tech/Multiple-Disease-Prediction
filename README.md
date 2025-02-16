# Multiple-Disease-Prediction
## Heart Disease Prediction  

## Overview  
This project implements a heart disease prediction model using Logistic Regression. The model is trained on a dataset sourced from Kaggle, which contains patient health records with various medical attributes.  

## Dataset  
The dataset includes various medical features related to heart health.  

### Attribute Information  
- **AGE**: Age in years  
- **SEX**: (1 = male; 0 = female)  
- **CP (Chest Pain Type)**:  
  - 0: Typical angina (most serious)  
  - 1: Atypical angina  
  - 2: Non-anginal pain  
  - 3: Asymptomatic (least serious)  
- **TRESTBPS**: Resting blood pressure (in mm Hg on admission to the hospital)  
- **CHOL**: Serum cholesterol in mg/dl  
- **FBS**: (Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  
  - A fasting blood sugar level less than 100 mg/dL is normal.  
  - From 100 to 120 mg/dL is considered prediabetes.  
  - If it is 125 mg/dL or higher on two separate tests, it indicates diabetes.  
- **RESTECG (Resting Electrocardiographic Results)**:  
  - 0: Normal  
  - 1: ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)  
  - 2: Probable or definite left ventricular hypertrophy by Estes' criteria  
- **THALACH**: Maximum heart rate achieved  
- **EXANG**: Exercise-induced angina (1 = yes; 0 = no)  
- **OLDPEAK**: ST depression induced by exercise relative to rest  
- **SLOPE (The slope of the peak exercise ST segment)**:  
  - 0: Upsloping  
  - 1: Flat  
  - 2: Downsloping  
- **CA**: Number of major vessels (0-3) colored by fluoroscopy  
- **THAL**:  
  - 3 = Normal  
  - 6 = Fixed defect  
  - 7 = Reversible defect  
- **TARGET (Diagnosis of heart disease - angiographic disease status)**:  
  - 0: < 50% diameter narrowing (No heart disease)  
  - 1: > 50% diameter narrowing (Heart disease present)  

## Model  
- **Algorithm:** Logistic Regression  
- **Accuracy:** 80%  

## Installation  
1. Clone the repository:  
   ```bash
   git clone 
