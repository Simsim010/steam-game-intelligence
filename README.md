#  Steam Game Intelligence

A machine learning project that analyzes Steam game reviews using sentiment analysis.

##  Live Demo

Try the application here:

👉 https://steam-game-intelligence.streamlit.app/

##  About the Project

This project analyzes Steam user reviews using **TF-IDF** and **Logistic Regression**.

The application allows users to:

-  Predict whether a Steam review is positive or negative.
-  Analyze game-level review statistics.
-  Explore model performance.
-  Examine prediction errors.

##  Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 92.22% |
| Precision | 93.74% |
| Recall | 96.96% |
| F1 Score | 95.32% |
| ROC-AUC | 0.9593 |

##  Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## ▶ Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py