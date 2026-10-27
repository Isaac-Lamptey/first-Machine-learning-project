# Classical ML Sprint: Diabetes Risk Classification

A self-directed learning project. I built logistic regression from scratch in NumPy, then moved to scikit-learn and compared models on a 15,000-row diabetes-risk dataset. The goal was to understand what the libraries do underneath, and to learn how to evaluate a model honestly.

> **This is a learning project, not a clinical tool.** Do not use it to make medical decisions.

**Author:** Isaac Lamptey, Computer Science & Engineering student, UMAT (Ghana)

## The problem

Predict whether a patient is **High risk** for diabetes (1) or not (0), from health and lifestyle columns.

- **Data:** `diabetes_risk.csv`, 15,000 rows, 19 columns. Source: [Kaggle datasets]
- **Class balance:** about 15% High risk, 85% not. This imbalance shapes everything below.
- **Target:** `diabetes_risk == "High"`

## What's in this repo

| File | What it does |
|---|---|
| `explore.py`, `pandas_basics.py`, `Numpy_basics.py` | pandas and NumPy practice: loading, filtering, groupby, plots |
| `logreg_scratch.py` | Logistic regression from scratch: sigmoid, cross-entropy loss, gradients, training loop, confusion matrix |
| `sklearn_logreg.py` | The same problem with scikit-learn, using a train/test split; saves the model with joblib |
| `tree_forest.py` | Decision Tree vs Random Forest |
| `depth_sweep.py` | Overfitting by `max_depth`, 5-fold cross-validation, threshold tuning, final test-set check |
| `categoricals.py` | Adds the text columns using one-hot encoding |
| `cv_and_threshold_tables.xlsx` | CV and threshold tables with a cost scenario you can edit |

## Results

| Experiment | Result |
|---|---|
| From-scratch logistic regression, **unscaled** features | Loss stalled near 0.44 after 1,000 epochs |
| Same code, **standardized** features | Loss 0.28 after 1,000 epochs, about 0.21 after 3,000 |
| From-scratch model, default 0.5 threshold (evaluated on training data) | Accuracy 0.92, but **recall only 0.56** |
| scikit-learn logistic regression, held-out test set | Accuracy 0.93, precision 0.81, recall 0.65, F1 0.72 |
| Decision Tree vs Random Forest (5 numeric features, test set) | F1 0.68 vs 0.75 |
| Random Forest CV F1 by `max_depth` | depth 2: 0.688, 3: 0.720, 5: 0.741, 10: 0.744, None: 0.740 (5, 10 and None are statistically tied, so depth 5 was chosen) |
| Threshold 0.2 vs 0.5, held-out test set | 0.2: precision 0.60, recall 0.89. 0.5: precision 0.87, recall 0.70 |
| Remove the two lab columns (`hba1c_level`, `fasting_blood_sugar`) | At threshold 0.2, F1 fell from about 0.72 to about 0.35 |
| Add all text columns (51 features after one-hot encoding) | CV F1 0.754, no meaningful gain over the numeric features |

## What I learned

1. **Scale your features.** Unscaled inputs made gradient descent race down one direction and crawl along the other. Standardizing fixed it with no other change.
2. **Accuracy can hide failure.** With 15% positives, a model can look good on accuracy while missing nearly half of the real positives. Recall exposed it.
3. **Choose thresholds on training data.** I picked the threshold with cross-validated probabilities from the training set, then touched the test set once. Tuning on the test set would have made the numbers optimistic.
4. **The best threshold depends on cost.** A missed High-risk patient and a false alarm cost different amounts. The spreadsheet shows the best threshold moving as that ratio changes.
5. **A strong feature can be a warning sign.** Almost all the predictive power sat in two lab values doctors use to diagnose diabetes. I can't tell whether the label was derived from them, and a screening tool would not have those values at prediction time.

## Limitations

- One train/test split with about 430 positive test cases, so recall is only good to about ±0.03.
- The from-scratch model was evaluated on its own training data.
- The dataset's origin and label definition are not verified.


## Run it

```bash
python -m venv venv
venv\Scripts\activate        # Windows
venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python logreg_scratch.py
python depth_sweep.py
```

## Deploy

## Use the API

```bash
pip install fastapi uvicorn scikit-learn pandas joblib
uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs, choose `POST /predict`, and press "Try it out".

Example request body:

```json
{"bmi": 29.0, "hba1c_level": 7.5, "age": 52, "fasting_blood_sugar": 130, "waist_circumference_cm": 102}
```

Example response:

```json
{"probability": 0.203, "flagged": true, "threshold": 0.2}
```

Inputs outside the training range are rejected with a 422 error, because the model can't extrapolate beyond data it has seen. Accepted ranges: BMI 15 to 40.8, HbA1c 4.8 to 13.4, age 18 to 80, waist 73.2 to 148.2 cm, fasting blood sugar 71 to 400.0 mg/dL.

The threshold is set to 0.2 to favour catching high-risk patients over avoiding false alarms. See the results table above. This is a learning project, not a clinical tool.

