# 🏥 Hospital Readmission Prediction System

![Hospital Readmission](https://img.shields.io/badge/Healthcare-Predictive%20Analytics-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![DEPI](https://img.shields.io/badge/DEPI-Graduation%20Project-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/header_image.png" alt="Hospital Readmission Prediction System" width="800"/>
</p>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Project Architecture](#-project-architecture)
- [Directory Structure](#-directory-structure)
- [Technologies & Tools](#-technologies--tools)
- [ML Model Development](#-ml-model-development)
- [Installation & Setup](#-installation--setup)
- [Web Application](#-web-application)
- [Database Schema](#-database-schema)
- [Deployment Options](#-deployment-options)
- [Future Enhancements](#-future-enhancements)
- [Contributors](#-contributors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🔍 Project Overview

The Hospital Readmission Prediction System is a comprehensive end-to-end machine learning solution developed as a graduation project for the DEPI program. This system predicts the likelihood of diabetic patients being readmitted to the hospital within 30 days after discharge, enabling healthcare providers to implement targeted interventions for high-risk patients.

Healthcare facilities face significant challenges with patient readmissions, which increase costs and reflect potential gaps in care quality. Our system addresses this problem by leveraging machine learning algorithms to identify patients at high risk of readmission, allowing for proactive care management.

## 🌟 Key Features

- **Predictive Analytics**: Advanced Random Forest classifier to predict hospital readmission risk
- **Interactive Web Interface**: User-friendly dashboard for healthcare providers to input patient data
- **Real-time Predictions**: Instant risk assessment with visualized probability scores
- **Database Integration**: Secure SQLite database for storing patient records and prediction history
- **Data Visualization**: Comprehensive charts and graphs for exploratory data analysis
- **Explainable AI**: Feature importance visualization to understand prediction factors
- **Responsive Design**: Mobile-friendly application accessible across devices

## 🎬 Demo

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/demo.gif" alt="Application Demo" width="700"/>
</p>

### Prediction Interface

Our web interface provides an intuitive form where healthcare providers can input patient information and receive instant predictions:

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/prediction_interface.png" alt="Prediction Interface" width="600"/>
</p>

### Results Dashboard

The results dashboard visually represents the prediction outcome with risk level indicators and key contributing factors:

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/results_dashboard.png" alt="Results Dashboard" width="600"/>
</p>

## 🏗️ Project Architecture

The system follows a three-tier architecture:

1. **Presentation Layer**: HTML/CSS frontend interface
2. **Application Layer**: Flask/Streamlit server handling requests and predictions
3. **Data Layer**: SQLite database and trained ML model

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/architecture.png" alt="System Architecture" width="700"/>
</p>

## 📁 Directory Structure

```
DEPI-Grad-Project/
│
├── app/
│   ├── app.py                      # Main Flask application
│   ├── app1.py                     # Alternate Streamlit application
│   ├── static/
│   │   ├── css/                    # CSS stylesheets
│   │   ├── js/                     # JavaScript files
│   │   └── images/                 # UI images and icons
│   └── templates/
│       ├── index.html              # Main prediction interface
│       └── results.html            # Results display page
│
├── models/
│   ├── random_forest_model.pkl     # Trained Random Forest model
│   └── model_metrics.json          # Model performance metrics
│
├── data/
│   ├── patients.db                 # SQLite database with patient records
│   ├── raw/                        # Raw input data
│   └── processed/                  # Processed datasets
│
├── notebooks/
│   ├── prediction-on-hospital-readmission.ipynb  # EDA & model training
│   └── feature_engineering.ipynb   # Feature engineering experiments
│
├── utils/
│   ├── data_preprocessing.py       # Data preprocessing utilities
│   ├── model_training.py           # Model training scripts
│   └── db_operations.py            # Database operation functions
│
├── tests/
│   ├── test_app.py                 # App functionality tests
│   └── test_model.py               # Model performance tests
│
├── requirements.txt                # Project dependencies
├── Dockerfile                      # Docker configuration
├── Procfile                        # Heroku deployment configuration
├── setup.py                        # Package setup script
└── README.md                       # Project documentation
```

## 🛠️ Technologies & Tools

### Backend
- **Python 3.8+**: Core programming language
- **Flask/Streamlit**: Web application frameworks
- **SQLite**: Lightweight database for patient records
- **Scikit-learn**: Machine learning library for model development
- **Pandas & NumPy**: Data manipulation and analysis
- **Joblib**: Model serialization and persistence

### Frontend
- **HTML5/CSS3**: Structure and styling
- **JavaScript/jQuery**: Interactive elements
- **Bootstrap 5**: Responsive design framework
- **Chart.js**: Data visualization library

### DevOps & Deployment
- **Git**: Version control system
- **Docker**: Containerization
- **Heroku**: Cloud platform deployment
- **GitHub Actions**: CI/CD pipeline

## 🧠 ML Model Development

Our machine learning pipeline includes:

### Data Preprocessing
- Handling missing values with appropriate imputation strategies
- Feature engineering to derive new meaningful features
- Encoding categorical variables
- Scaling numerical features
- Addressing class imbalance with SMOTE

### Model Selection
We evaluated multiple algorithms including:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machines

### Performance Metrics
Random Forest was selected as our final model based on:
- ROC AUC: 0.87
- Precision: 0.83
- Recall: 0.76
- F1 Score: 0.79

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/model_performance.png" alt="Model Performance Metrics" width="600"/>
</p>

### Feature Importance

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/DEPI-Grad-Project/main/assets/feature_importance.png" alt="Feature Importance" width="600"/>
</p>

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Zoad-Henedy2/DEPI-Grad-Project.git
cd DEPI-Grad-Project
```

### Step 2: Create and Activate Virtual Environment (Optional)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up the Database
```bash
python utils/db_setup.py
```

### Step 5: Run the Application
#### For Flask App:
```bash
python app/app.py
```
#### For Streamlit App:
```bash
streamlit run app/app1.py
```

### Step 6: Access the Web Interface
Open your browser and go to:
- Flask: `http://localhost:5000`
- Streamlit: `http://localhost:8501`

## 📱 Web Application

The web application offers several key features:

### Patient Information Input Form
- Demographic data (age, gender, race)
- Medical history (diagnoses, medications)
- Lab results and vital signs
- Previous admission details

### Prediction Dashboard
- Risk score visualization
- Color-coded risk levels (Low, Medium, High)
- Confidence intervals for predictions
- Top contributing factors

### Admin Panel
- Model performance monitoring
- Database management
- User access control
- Audit logs for prediction history

## 🗄️ Database Schema

Our SQLite database (`patients.db`) contains the following key tables:

### Patients Table
```sql
CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT,
    race TEXT,
    admission_type INTEGER,
    discharge_disposition INTEGER,
    admission_source INTEGER,
    time_in_hospital INTEGER,
    num_lab_procedures INTEGER,
    num_medications INTEGER,
    num_diagnoses INTEGER,
    max_glu_serum TEXT,
    A1Cresult TEXT,
    insulin TEXT,
    diabetesMed TEXT,
    readmitted TEXT
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    prediction_score REAL,
    prediction_date TEXT,
    actual_outcome TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
```

## 🚢 Deployment Options

### Docker Deployment
```bash
# Build Docker image
docker build -t hospital-readmission .

# Run container
docker run -p 5000:5000 hospital-readmission
```

### Heroku Deployment
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create hospital-readmission-predictor

# Push to Heroku
git push heroku main

# Open the deployed app
heroku open
```

### Local Deployment with Gunicorn (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

## 🔮 Future Enhancements

- **Model Improvements**: Implement deep learning models for higher accuracy
- **API Development**: Create RESTful API for integration with hospital EMR systems
- **Advanced Visualizations**: Add interactive dashboards with Plotly/Dash
- **Mobile Application**: Develop companion mobile app for on-the-go predictions
- **Real-time Monitoring**: Implement real-time model performance tracking
- **Multi-language Support**: Add localization for global healthcare providers

## 👥 Contributors

- **Ziad Henedy** - _Project Lead & ML Engineer_ - [GitHub Profile](https://github.com/Zoad-Henedy2)
- _Add other team members here_

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DEPI program faculty and mentors
- Healthcare professionals who provided domain expertise
- Open-source machine learning community
- Dataset providers for making healthcare data accessible for research

---

<p align="center">
  <b>Developed with ❤️ for improving healthcare outcomes</b>
</p>

<p align="center">
  <i>For questions or feedback, please create an issue or contact us directly.</i>
</p>
