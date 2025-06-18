
# 🏥 Patient Readmission Prediction

This project uses the **Diabetes 130-US hospitals for years 1999–2008** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) to build a machine learning model that predicts **whether a patient will be readmitted within 30 days** after discharge.

## 📂 Project Structure

```
├── Readmission.ipynb          # Main Jupyter notebook for EDA, preprocessing, training, and evaluation
├── mapped_data.csv            # Cleaned and preprocessed version of the dataset
├── random_forest_model.pkl    # Trained Random Forest model saved for inference
└── README.md                  # Project documentation (this file)
```

## 📊 Dataset

- **Source**: UCI Machine Learning Repository  
- **Original Dataset**: [Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296)
- **Size**: Over 100,000 hospital records for diabetic patients
- **Target**: `readmitted` (`<30`, `>30`, `NO`), converted to a binary classification (`<30` = readmitted, other = not readmitted)

## ⚙️ Problem Statement

Predict if a diabetic patient will be readmitted to the hospital within 30 days of discharge based on:

- Demographic features (e.g., race, gender, age)
- Hospitalization details (e.g., number of procedures, length of stay)
- Medication and diagnosis data

## 🔍 Approach

1. **Data Cleaning**:
   - Removed columns with too many missing values or ID-like information.
   - Mapped categorical features into meaningful labels (e.g., `age` ranges → numerical midpoints).

2. **Feature Engineering**:
   - One-hot encoding of nominal variables.
   - Ordinal conversion for ordered categories (like age).

3. **Modeling**:
   - Random Forest Classifier
   - Trained on balanced classes to reduce bias from class imbalance.
   - Saved as `random_forest_model.pkl` for reuse.

4. **Evaluation**:
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix
   - ROC AUC Curve

## 📈 RandomForest Model Performance

| Metric        | Value     |
|---------------|-----------|
| Accuracy      |  `59.7%`    |
| Precision     |  `17.4%`    |
| Recall        |  `64.3%`    |
| F1 Score      |  `27.4`     |
| AUC-ROC       |  `61.7`     |

## 📈 NN Model Performance

| Metric        | Value     |
|---------------|-----------|
| Accuracy      |  `79.2%`    |
| Precision     |  `22.7%`    |
| Recall        |  `31.5%`    |
| F1 Score      |  `26.4`     |

## 🧪 How to Run

1. Clone the repo or download the files.
2. Open `Readmission.ipynb` in Jupyter Notebook.
3. Run all cells step-by-step:
   - Data loading
   - Preprocessing
   - Training
   - Evaluation
4. To use the saved model:
   ```python
   import joblib
   model = joblib.load('random_forest_model.pkl')
   prediction = model.predict(X_new)
   ```

## 📌 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn / matplotlib (for visualization)
- jupyter

You can install dependencies using:

```bash
pip install -r requirements.txt
```

(You may need to create the `requirements.txt` by exporting your environment.)

## 📚 References

- [UCI Diabetes Dataset](https://archive.ics.uci.edu/dataset/296)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Paper on Dataset](https://www.hindawi.com/journals/bmri/2014/781670/)

## 📬 Contact

If you have questions or suggestions, feel free to reach out!
