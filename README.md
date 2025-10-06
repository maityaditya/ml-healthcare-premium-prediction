# ml-healthcare-premium-prediction
# 🩺 ML Healthcare Premium Prediction

A **Machine Learning-powered Streamlit web app** that predicts **Healthcare Insurance Premiums** based on personal and lifestyle factors like age, gender, BMI, smoking status, and region.

🔗 **Live Demo:** [Healthcare Premium Prediction App](https://machine-learning-healthcare-premium-prediction.streamlit.app/)  
💻 **GitHub Repository:** [ml-healthcare-premium-prediction](https://github.com/maityaditya/ml-healthcare-premium-prediction)

---

## 📘 Project Overview

This project demonstrates how **Machine Learning** can be applied in the healthcare insurance domain to estimate the **insurance premium** of a customer.

The app:
- Accepts user input (Age, Gender, BMI, Children, Smoking Status, Region)
- Preprocesses and encodes data
- Predicts the healthcare premium using a trained regression model
- Displays results interactively through a **Streamlit** interface

---

## 🧠 Objective

The main goal is to build an interactive ML-based system that helps predict **insurance premium costs** using user details.  
It provides a practical example of deploying a machine learning model in a real-world scenario using **Streamlit**.

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Programming Language** | Python 🐍 |
| **Frontend / Deployment** | Streamlit |
| **Model Development** | Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git & GitHub |

---
## 📁 Project Structure
ml-healthcare-premium-prediction/

- **artifacts/** – Stored trained model and preprocessing artifacts  
- **main.py** – Main Streamlit app file  
- **prediction_helper.py** – Helper functions for feature preprocessing and prediction  
- **requirements.txt** – List of required Python libraries  
- **LICENSE** – License file (Apache 2.0)  
- **README.md** – Project documentation


---

## 🚀 How to Run Locally

Follow these simple steps to run the app on your local system:

### 1️⃣ Clone the repository

bash
git clone https://github.com/maityaditya/ml-healthcare-premium-prediction.git
cd ml-healthcare-premium-prediction

2️⃣ Install dependencies

Make sure you have Python 3.8+ installed, then install the required libraries:

pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run main.py


Once the server starts, Streamlit will open automatically in your browser.
If not, visit 👉 http://localhost:8501

🧩 Features

✅ Simple and clean Streamlit user interface
✅ Real-time premium prediction based on user inputs
✅ Encodes categorical features (Gender, Smoker, Region) automatically
✅ Includes saved ML model for instant prediction (no retraining needed)
✅ Ready-to-deploy and customizable for new datasets

📊 Model Details

The app uses a Regression Model trained on healthcare insurance data with the following key features:

Feature	Description
Age	Age of the individual
Gender	Male / Female
BMI	Body Mass Index
Children	Number of dependents
Smoker	Yes / No
Region	Geographic area
Example Algorithms Used:

Linear Regression

Random Forest Regressor

XGBoost Regressor (optional for better accuracy)

You can update prediction_helper.py with your preferred model or tuning configuration.

🧪 Example Usage

Open the deployed app or run locally.

Enter the details (e.g., Age = 35, BMI = 27.5, Smoker = Yes).

Click Predict Premium.

The model instantly displays the predicted insurance premium.

📈 Future Enhancements

🔹 Add advanced models (XGBoost, LightGBM)
🔹 Improve feature engineering & scaling
🔹 Add SHAP/LIME-based model interpretability
🔹 Enhance frontend design with better visuals
🔹 Integrate with a database for storing predictions
🔹 Deploy using Docker or CI/CD pipeline

🧑‍💻 Author

Aditya Maity
🎓 BTech CSE | Data Science & ML Enthusiast
🔗 GitHub Profile

🪪 License

This project is licensed under the Apache 2.0 License.
You are free to use, modify, and distribute this software with proper attribution.

⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub
!
It helps others find and use it too 😊

🩺 “Machine Learning meets Healthcare — predict smarter, live healthier!”


