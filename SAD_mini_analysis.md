# SAD Mini-Analysis: Student Performance Prediction System

**Name(s):** ______________________  **Section:** ____________  **Date:** ____________

| SAD element | Response |
| --- | --- |
| **Problem statement** | Academic risk is often noticed only after poor results are recorded. Teachers lack an early, data-driven signal of which learners may need additional academic support. The prototype explores early identification of students likely to fail the final Mathematics grade using indicators that are already available. |
| **Users** | Faculty member or academic analyst who reviews risk flags and decides on support. |
| **Inputs** | Student attributes and academic indicators: study time, absences, past failures, G1 and G2 period grades, family and social factors. Supplied as a CSV file with the same predictor columns as the training data (G3 excluded). |
| **Process** | Validate uploaded data (required columns check) → preprocess (impute missing values, scale numeric, one-hot encode categorical) → apply trained Random Forest model → classify risk. |
| **Outputs** | AT RISK / NOT AT RISK label per student, count of predicted at-risk records, downloadable results CSV, and a model evaluation summary (accuracy, precision, recall, F1-score). |
| **Key limitation** | A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population. Predictions are not diagnoses and must not be the sole basis for real student decisions. |

## Mini use-case
The user uploads a CSV file → the system checks required columns → the model predicts risk → the system displays and exports the results.

## Requirements (summary)
- **Functional:** upload CSV; validate columns; predict risk; display and export results; show dataset summary.
- **Non-functional:** runs locally; clear error messages; no personal data stored beyond the session.

## SAD stages mapping
| Stage | Where in the prototype |
| --- | --- |
| Problem | Problem statement above |
| Requirements | Inputs, outputs, requirements summary |
| Design | Process flow; pipeline in `train_model.py`; UI in `app.py` |
| Implementation | Python, scikit-learn, Streamlit, GitHub |
| Testing | Test table in Section 10 of the lab handout |
