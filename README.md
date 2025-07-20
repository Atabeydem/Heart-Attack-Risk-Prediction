# ❤️ Heart Attack Risk Prediction

This project predicts the risk of heart attack based on clinical input parameters using a machine learning model. It provides two main interfaces:

- 🖥️ **Streamlit Frontend** for real-time prediction
- ⚙️ **FastAPI Backend** served via Docker

---

## 🧠 Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to assist early diagnosis by providing an AI-powered risk prediction tool trained on clinical features.

- Input: Clinical measurements such as age, cholesterol, blood pressure, etc.
- Output: Risk class (0 or 1) and probability (e.g., “36.39%”)
- Model: Random Forest (or configurable alternatives)
- Technologies: Python, scikit-learn, FastAPI, Streamlit, Docker

---

## 🗂️ Folder Structure

```
Heart-Attack-Risk-Prediction/
├── models/                 # Trained model files (.joblib)
├── streamlit_app.py        # Interactive frontend
├── fastapi_app.py          # Backend API with FastAPI
├── Dockerfile              # Docker config for API
├── requirements.txt        # Python dependencies
├── .dockerignore           # Docker ignore rules
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart-Attack-Risk-Prediction.git
cd Heart-Attack-Risk-Prediction
```

### 2. Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This will launch an interactive UI in your browser to enter clinical data and get predictions.

---

## 🔧 Run the FastAPI Server with Docker

> Make sure Docker is installed and running.

### 1. Build the Docker Image

```bash
docker build -t heart-api .
```

### 2. Run the Container

```bash
docker run -p 8000:8000 -v "${PWD}/models:/app/models" heart-api
```

### 3. Access the API

- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Note: The API documentation is available at http://localhost:8000/docs when the server is running locally.
- Example prediction (via `/predict` endpoint):

```json
{
  "Age": 59,
  "Sex": 1,
  "ChestPainType": 2,
  "RestingBP": 130,
  "Cholesterol": 250,
  "FastingBS": 0,
  "RestingECG": 1,
  "MaxHR": 145,
  "ExerciseAngina": 0,
  "Oldpeak": 1.0,
  "ST_Slope": 2
}
```

Response:
```json
{
  "prediction": 0,
  "risk_probability": "36.39%"
}
```

---

## 📦 Dependencies

- Python 3.10
- scikit-learn
- joblib
- streamlit
- fastapi
- uvicorn

---

## 🧩 Future Improvements

- Add user authentication (JWT)
- Deploy to Render / AWS / Azure
- Model monitoring and explainability (e.g., SHAP)
- Unit testing and CI/CD pipelines

---

## 📄 License

MIT License — feel free to use and modify this project.
