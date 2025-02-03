# Titanic Survival Prediction
# Task: Train and evaluate Logistic Regression, Random Forest, and Neural Network models.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# data_path = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data_path = 'titanic_dataset.csv'
df = pd.read_csv(data_path) # loading the dataset into a pandas dataframe from a csv file or directly form the online source 

# Handling the missing values i.e. preprocessing the data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# adjusting/cropping out data by keeping the columns required and dropping the others
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# converting categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Splitting data into features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Splitting the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainning different machine learning models (Logistic Regression, Random Forest and Neural Network as indicated in the task)
# Trainning Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Trainning Random Forest model
random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest.fit(X_train, y_train)
rf_preds = random_forest.predict(X_test)

# Trainning Neural Network model
nn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Converting labels to numpy arrays
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Making predictions and comparision is then followed
nn_preds = (nn_model.predict(X_test) > 0.5).astype("int32").flatten()

# Creating a function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"\nPerformance Metrics for {model_name}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

# Calling the evaluate_model function and supplying the respective prediction data from all models
evaluate_model(y_test, log_reg_preds, "Logistic Regression")
evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, nn_preds, "Neural Network")

# Summarizing the overall performance
results = {
    "Model": ["Logistic Regression", "Random Forest", "Neural Network"],
    "Accuracy": [accuracy_score(y_test, log_reg_preds), accuracy_score(y_test, rf_preds), accuracy_score(y_test, nn_preds)],
    "Precision": [precision_score(y_test, log_reg_preds), precision_score(y_test, rf_preds), precision_score(y_test, nn_preds)],
    "Recall": [recall_score(y_test, log_reg_preds), recall_score(y_test, rf_preds), recall_score(y_test, nn_preds)],
    "F1 Score": [f1_score(y_test, log_reg_preds), f1_score(y_test, rf_preds), f1_score(y_test, nn_preds)],
}

summary_df = pd.DataFrame(results)
print("\nSummary of Performance Metrics:")
print(summary_df)
