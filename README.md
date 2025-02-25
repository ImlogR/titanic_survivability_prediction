# titanic_survivability_prediction

This project implements a **Supervised Machine Learning Model** to predict the survival of Titanic passengers. It compares three models:
- **Logistic Regression**
- **Random Forest**
- **Neural Network (TensorFlow/PyTorch)**

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [License](#license)

## Installation
To run this project, install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow torch
```
Or Simply run:
```
pip install -r requirements.txt
```

## Dataset
The dataset contains information about Titanic passengers, including:
- **PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked**

## Preprocessing
1. **Handle Missing Values:**
   - Fill missing `Age` values with the median.
   - Fill missing `Embarked` values with the mode.
   - Fill missing `Fare` values with the median.

2. **Drop Unnecessary Columns:**
   - Remove `PassengerId`, `Name`, `Ticket`, and `Cabin` as they don't contribute to survival prediction.

3. **Convert Categorical Variables:**
   - Use **One-Hot Encoding** on `Sex` and `Embarked`.
   - Drop the first dummy variable to avoid the **dummy variable trap**.

4. **Split Dataset:**
   - `X` (features) and `y` (target variable) are separated.
   - Data is split into **80% training and 20% testing**.

## Model Training & Evaluation
We trained three models using `X_train` and `y_train`, then evaluated them on `X_test` and `y_test`:
1. **Logistic Regression**
2. **Random Forest**
3. **Neural Network (TensorFlow/PyTorch)**

### Evaluation Metrics:
- **Accuracy**: Overall correctness of predictions.
- **Precision**: How many predicted positives were actually correct.
- **Recall**: How many actual positives were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

## Results
| Model               | Accuracy | Precision | Recall  | F1-Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression | 0.798883 | 0.771429  | 0.729730| 0.750000 |
| Random Forest       | 0.821229 | 0.800000  | 0.756757| 0.777778 |
| Neural Network      | 0.754190 | 0.697368  | 0.716216| 0.706667 |

## Conclusion
- **Random Forest** may perform better than Logistic Regression due to its ability to handle non-linearity.
- **Neural Network** can achieve higher accuracy with proper tuning but requires more computation.
- Feature selection and hyperparameter tuning can further improve model performance.

## Usage
To run the project, execute:
```bash
python main.py
```

## License
This project is open-source and available under the MIT License.

