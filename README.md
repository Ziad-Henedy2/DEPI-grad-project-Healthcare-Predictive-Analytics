
# 🏥 Hospital Readmission Prediction - DEPI Graduation Project

This project is a complete end-to-end solution that predicts the likelihood of a patient being readmitted to the hospital using machine learning. It includes a web-based interface, a trained model, and exploratory analysis in a Jupyter notebook.

## 📁 Project Structure

```
DEPI-Grad-Project/
│
├── app.py                        # Main Flask/Streamlit app
├── app1.py                       # Alternate version of the app
├── index.html                    # Frontend interface for prediction
├── random_forest_model.pkl       # Trained Random Forest model
├── patients.db                   # SQLite database with patient records
├── prediction-on-hospital-readmission.ipynb  # EDA + ML notebook
```

## 🚀 Features

- Predict readmission risk for diabetic patients
- Web interface to input patient data and get predictions
- Trained Random Forest classifier
- Exploratory Data Analysis and visualizations
- SQLite database integration

## 🧠 Technologies Used

- Python (Pandas, Scikit-learn, Joblib)
- Flask or Streamlit (for the app)
- HTML/CSS (frontend)
- SQLite (database)
- Jupyter Notebook (EDA & modeling)

## ▶️ How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/Zoad-Henedy2/DEPI-Grad-Project.git
cd DEPI-Grad-Project
```

2. **Install required libraries:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
python app.py
```

4. **Access the interface:**
Go to `http://localhost:5000` in your browser.

## 📊 Notebooks

The notebook `prediction-on-hospital-readmission.ipynb` includes:
- Data loading & cleaning
- Feature engineering
- Model training (Random Forest)
- Evaluation (ROC, precision/recall, confusion matrix)

## 📦 Deployment

This app can be deployed to:
- **Heroku** (Flask setup with `Procfile`)
- **Streamlit Sharing**
- **Docker** (create a Dockerfile and containerize)


## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

For questions or feedback, please reach out via GitHub Issues or email.
