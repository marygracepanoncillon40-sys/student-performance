# Student Performance Prediction System

A classroom prototype that estimates whether a secondary-school student is at risk of failing the final Mathematics grade, with a Streamlit app for CSV upload and prediction.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification using available student indicators (study time, absences, past failures, G1 and G2 grades, and others).

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320): https://archive.ics.uci.edu/dataset/320/student+performance
The Mathematics file (`student-mat.csv`, 395 records) is used.

## Prediction target
`G3 < 10` is **AT RISK**; `G3 >= 10` is **NOT AT RISK**. `G3` is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py        # downloads data, trains model, creates student_data.csv and student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions. Results can be downloaded as CSV.

## Model results
Fill in after running `train_model.py`:

| Metric (At risk class) | Value |
| --- | --- |
| Accuracy | |
| Precision | |
| Recall | |
| F1-score | |

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to another institution or population.
- G1 and G2 are strong predictors, so the model is most useful after early grading periods.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
