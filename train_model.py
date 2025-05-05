import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Generate synthetic training data
data_size = np.random.randint(10, 10000, 5000)
sortedness = np.random.choice([0, 1, 2], 5000)  # 0: sorted, 1: reverse_sorted, 2: random
duplicates = np.random.randint(0, 1000, 5000)

# Feature matrix
X = np.column_stack((data_size, sortedness, duplicates))

# Target: Selecting the best sorting algorithm dynamically
# 0 -> Insertion Sort, 1 -> Quick Sort, 2 -> Merge Sort (Can be expanded)
y = np.array([(0 if size < 100 else 1 if size < 1000 else 2) for size in data_size])  

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Neural Network Model (MLP)
model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', 
                      max_iter=500, random_state=42, learning_rate_init=0.001)

# Train the model
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "sorting_nn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Neural Network Model trained successfully!")

# Predict on train and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Insertion", "Quick", "Merge"], 
            yticklabels=["Insertion", "Quick", "Merge"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_test, y_test_pred))