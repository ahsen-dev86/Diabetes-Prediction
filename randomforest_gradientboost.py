import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data_path = "diabetes-data-set/diabetes.csv"
df = pd.read_csv(data_path)

# Features and target
X = df[['Age', 'BMI', 'Insulin', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction']]
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Starting Random Forest hyperparameter tuning...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print(f"Best RF params: {rf_grid.best_params_}")
rf_best = rf_grid.best_estimator_

# Predict and evaluate Random Forest best model
rf_pred = rf_best.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy: {rf_acc:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred))

# Gradient Boosting Classifier tuning
gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

print("\nStarting Gradient Boosting hyperparameter tuning...")
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3, n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train)

print(f"Best GB params: {gb_grid.best_params_}")
gb_best = gb_grid.best_estimator_

# Predict and evaluate Gradient Boosting best model
gb_pred = gb_best.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

print(f"\nGradient Boosting Accuracy: {gb_acc:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_pred))
print("Classification Report:")
print(classification_report(y_test, gb_pred))
