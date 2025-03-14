import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving models

# Load dataset
df = pd.read_csv("balanced_tf_idf_undersampling_dataset.csv")

# Separate features and target
X = df.drop(columns=['Label'])
y = df['Label']

# Split dataset: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear')  # Using linear kernel for better interpretability
}

# Train and evaluate models
best_model = None
best_acc = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"📊 {name} Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = (name, model)

# Evaluate the best model on the test set
best_name, best_clf = best_model
y_pred_test = best_clf.predict(X_test)
print(f"\n✅ Best Model: {best_name} - Test Set Performance ✅")
print(classification_report(y_test, y_pred_test))

# Save the best model
joblib.dump(best_clf, f"{best_name.lower().replace(' ', '_')}_model.pkl")
print(f"💾 Model saved as {best_name.lower().replace(' ', '_')}_model.pkl")




