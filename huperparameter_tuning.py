import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("balanced_tf_idf_undersampling_dataset.csv")

# Separate features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Split into 80% training, 10% validation, 10% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define parameter grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs', 'saga'],  # Optimization solvers
    'penalty': ['l1', 'l2']  # Regularization type
}

# Initialize and perform GridSearch
log_reg = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and validation accuracy
best_model = grid_search.best_estimator_
print(f"🏆 Best Parameters: {grid_search.best_params_}")
print(f"📊 Best Validation Accuracy: {grid_search.best_score_:.4f}")

# Evaluate best model on test set
y_pred_test = best_model.predict(X_test)
print("\n✅ Final Test Set Performance:")
print(classification_report(y_test, y_pred_test))

# Save best model
import joblib
joblib.dump(best_model, "tuned_logistic_regression_model.pkl")
print("\n💾 Best Tuned Model saved as 'tuned_logistic_regression_model.pkl'")
