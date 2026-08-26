# Machine Learning System for Equipment Failure

> An AI-powered predictive maintenance system that predicts equipment failure using industrial sensor data and provides real-time machine health, risk assessment, alerts, and maintenance recommendations through an interactive Streamlit dashboard.

---

## 🚀 Live Demo

**Web Application:**  
https://ai-smart-predictive-maintenance-hdhp5urxg4umpdqertsaxz.streamlit.app/

---

## 📌 About the Project

Equipment failures can cause unexpected downtime, production losses, and increased maintenance costs. Traditional maintenance approaches often depend on fixed schedules or respond only after a failure occurs.

This project uses **Machine Learning for Predictive Maintenance** to identify the possibility of equipment failure based on operational and sensor parameters.

A **Random Forest Classifier** is trained using the **AI4I 2020 Predictive Maintenance Dataset**. The trained model is integrated into a **Streamlit web application**, where users can enter equipment parameters and receive an immediate failure prediction along with a machine health score, failure probability, risk level, and recommended maintenance action.

The system also provides an analytics dashboard and maintains a history of previous predictions.

---

## 🎯 Objectives

- Predict the possibility of equipment failure using machine learning.
- Analyze important industrial equipment parameters.
- Provide real-time failure probability and machine health assessment.
- Classify equipment conditions into different risk levels.
- Provide maintenance recommendations based on predicted risk.
- Visualize equipment and failure-related data.
- Maintain a history of previous predictions.
- Deploy the complete system as a web application.

---

## 🧠 Machine Learning Model

### Random Forest Classifier

The project uses a **Random Forest Classifier** for equipment failure prediction.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to make a final prediction. It is suitable for this problem because equipment failure can depend on complex relationships between multiple sensor and operational parameters.

The trained model and preprocessing scaler are saved using **Joblib** and loaded by the Streamlit application during prediction.

### Prediction Features

The model uses the following equipment parameters:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains simulated industrial equipment data and includes operational parameters and machine failure information.

### Important Parameters

| Parameter | Description |
|---|---|
| Machine Type | Equipment variant |
| Air Temperature | Ambient air temperature |
| Process Temperature | Machine process temperature |
| Rotational Speed | Rotational speed in RPM |
| Torque | Applied torque in Nm |
| Tool Wear | Tool usage time in minutes |
| Machine Failure | Target variable indicating failure |

### Failure Modes

The dataset also contains indicators for different failure conditions:

- **TWF** – Tool Wear Failure
- **HDF** – Heat Dissipation Failure
- **PWF** – Power Failure
- **OSF** – Overstrain Failure
- **RNF** – Random Failure

---

## 🖥️ Application Features

### 🏠 Home Dashboard

The Home page provides an overview of the predictive maintenance system.

It includes:

- System and model overview
- Equipment monitoring information
- Operational data visualization
- Machine failure monitoring
- Predictive maintenance capabilities

---

### 🔍 Equipment Failure Prediction

Users can enter equipment parameters through the dashboard.

The system then generates:

- Failure prediction
- Failure probability
- Machine health score
- Risk classification
- Maintenance recommendation

---

### 🚨 Risk & Alert System

The system categorizes equipment conditions according to the predicted failure probability.

| Risk Level | Condition | Action |
|---|---|---|
| 🟢 Low Risk | Low probability of failure | Continue normal operation |
| 🟡 Medium Risk | Increased failure probability | Schedule preventive maintenance |
| 🔴 High Risk | High probability of failure | Immediate inspection/intervention |

The dashboard provides visual alerts when the equipment reaches warning or critical risk levels.

---

### 📊 Analytics Dashboard

The Analytics page provides interactive visual analysis of the equipment dataset.

It includes:

- RPM vs Torque scatter plot
- Failure mode analysis
- Machine type distribution
- Tool wear distribution
- Sensor correlation matrix
- Dataset filtering
- Summary statistics

---

### 📜 Prediction History

The system records prediction results for future reference.

The prediction history contains:

- Date
- Time
- Machine Type
- Health Score
- Failure Probability
- Prediction Result
- Alert Level

The history can also be exported as a CSV file.

---

## 🔄 System Workflow

```text
                Industrial Dataset
                       │
                       ▼
               Data Preprocessing
                       │
                       ▼
                Feature Selection
                       │
                       ▼
                  Data Scaling
                       │
                       ▼
             Random Forest Training
                       │
                       ▼
                Model Evaluation
                       │
                       ▼
             Save Model & Scaler
                       │
                       ▼
              Streamlit Dashboard
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Sensor Input         Dataset Analytics
             │
             ▼
       ML Prediction
             │
             ▼
    Failure Probability
             │
             ▼
       Health & Risk
             │
             ▼
 Maintenance Recommendation
             │
             ▼
      Prediction History
