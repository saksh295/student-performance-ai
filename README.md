# 🎓 Student Performance Risk Predictor

An AI-powered machine learning application that predicts whether a student is **At Risk** or **Not At Risk** based on academic, family, and lifestyle factors.

The project uses a **Random Forest Classifier** trained on the UCI Student Performance dataset and provides an interactive prediction interface using **Streamlit**.

---

## 📌 Project Overview

Student academic performance can be influenced by several factors such as previous failures, study time, absences, grades, family support, and lifestyle habits.

This project uses machine learning to identify students who may be at risk of poor academic performance.

The application allows users to enter student information and receive:

* Risk prediction
* Prediction confidence
* Probability breakdown
* Basic recommendations based on the prediction

---

## 🎯 Objective

The main objective of this project is to develop a machine learning model that can classify students into two categories:

* **At Risk**
* **Not At Risk**

A student is considered **At Risk** when their final grade (`G3`) is below 10.

---

## 📊 Dataset

The project uses the **Student Performance dataset** containing information about students' academic performance, family background, study habits, and lifestyle.

The dataset contains **395 student records** and **33 original features**.

The target variable is created from the final grade:

```python
df["risk"] = df["G3"].apply(
    lambda x: "At Risk" if x < 10 else "Not At Risk"
)
```

---

## 🧠 Features Used

The model uses 18 features:

* Age
* Mother's education
* Father's education
* Travel time
* Study time
* Past class failures
* Extra educational support
* Family educational support
* Extra paid classes
* Extra-curricular activities
* Desire for higher education
* Internet access
* Free time
* Going out
* Health
* Absences
* First period grade (G1)
* Second period grade (G2)

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The project uses a Random Forest Classifier with:

* Number of estimators: **100**
* Random state: **42**
* Test size: **20%**
* Stratified train-test split

### Dataset Split

| Dataset  | Samples |
| -------- | ------: |
| Training |     316 |
| Testing  |      79 |
| Total    |     395 |

---

## 📈 Model Performance

The trained model achieved:

### **87.34% Accuracy**

Classification performance:

| Class       | Precision | Recall | F1-Score |
| ----------- | --------: | -----: | -------: |
| At Risk     |      0.75 |   0.92 |     0.83 |
| Not At Risk |      0.96 |   0.85 |     0.90 |

The model achieved particularly strong recall for the **At Risk** class, meaning it successfully identifies most students who are classified as being at risk.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit web application.

Users can enter:

### 📚 Academic Information

* Age
* Education levels
* Study time
* Travel time
* Previous failures
* Absences
* G1
* G2

### 🏠 Family & Support

* School support
* Family support
* Paid classes
* Higher education plans
* Extra-curricular activities
* Internet access

### 🌱 Lifestyle

* Free time
* Going out
* Health status

The application then predicts the student's risk category.

---

## 📊 Prediction Output

The application displays:

* **At Risk / Not At Risk**
* Prediction confidence
* Probability for each risk category
* Basic recommendations

Example:

```text
Prediction: Not At Risk
Confidence: 86%
```

---

## 📁 Project Structure

```text
student-performance-ai/
│
├── data/
│   └── student-mat.csv
│
├── models/
│   └── student_risk_model.pkl
│
├── notebooks/
│
├── src/
│   ├── data_analysis.py
│   └── model_training.py
│
├── venv/
│
├── app.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib
* Machine Learning
* Random Forest

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd student-performance-ai
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🔍 Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Analysis
   ↓
Create Risk Label
   ↓
Feature Selection
   ↓
Data Encoding
   ↓
Train/Test Split
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit Application
   ↓
Student Risk Prediction
```

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

The prediction should not be treated as a definitive assessment of a student's academic future. Actual academic performance can depend on many factors that may not be represented in the dataset.

---

## 👩‍💻 Author

**Sakshi Ingale**

BCA – Data Science
