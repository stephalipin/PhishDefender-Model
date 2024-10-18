import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from joblib import dump, load
import os

# File paths
legitimate_csv_path = r"./structured_data_legitimate.csv"
phishing_csv_path = r"./structured_data_phishing.csv"

# Read CSV files into dataframes
legitimate_df = pd.read_csv(legitimate_csv_path)
phishing_df = pd.read_csv(phishing_csv_path)

# Step 3 combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)

# Step 4 remove 'URL' and remove duplicates, then we can create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)
df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']

# Step 5 split data to train and test
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# Initialize classifiers
decision_tree = DecisionTreeClassifier(random_state=7)
random_forest = RandomForestClassifier(n_estimators=100, random_state=10)
xgboost = XGBClassifier(n_estimators=100, random_state=10)

# Train classifiers
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# Make predictions
dt_preds = decision_tree.predict(X_test)
rf_preds = random_forest.predict(X_test)
xgboost_preds = xgboost.predict(X_test)

# Ensemble predictions
ensemble_preds = []
for dt_pred, rf_pred, xgb_pred in zip(dt_preds, rf_preds, xgboost_preds):
    # Use a majority voting scheme
    if dt_pred + rf_pred + xgb_pred >= 2:
        ensemble_preds.append(1)  # Predict phishing
    else:
        ensemble_preds.append(0)  # Predict legitimate

# Calculate evaluation metrics
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
ensemble_precision = precision_score(y_test, ensemble_preds)
ensemble_recall = recall_score(y_test, ensemble_preds)
ensemble_f1 = f1_score(y_test, ensemble_preds)

print("PhishDefender")
print("Accuracy:", ensemble_accuracy)
print("Precision:", ensemble_precision)
print("Recall:", ensemble_recall)
print("F1 Score:", ensemble_f1)

# Plot performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color='orange')
plt.title('PhishDefender')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.show()

# Export ensemble model using joblib
ensemble_model = [decision_tree, random_forest, xgboost]
file_path = './model/phishdefender_model9.pkl'
dump(ensemble_model, file_path)

# Verify if the file is saved
if os.path.exists(file_path):
    print("Model successfully saved at:", file_path)
else:
    print("Error: Model not saved.")

