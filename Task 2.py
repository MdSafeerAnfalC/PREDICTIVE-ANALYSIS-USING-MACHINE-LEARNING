# Import necessary libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
# Step 1: Input the dataset 
# Sample dataset with features and target (for classification) 
data = pd.DataFrame({ 
'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
'feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
'feature3': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Binary classification target 
})
# Step 2: Preprocess the data 
X = data[['feature1', 'feature2', 'feature3']] # Features 
y = data['target'] # Target variable 
# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Step 3: Select a prediction model 
model = DecisionTreeClassifier() 
# Step 4: Train the model 
model.fit(X_train, y_train) 
# Step 5: Make predictions 
y_pred = model.predict(X_test) 
# Step 6: Evaluate the model 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model Accuracy: {accuracy}") 
# Step 7: Make a decision 
# Based on the predicted classes, we can take actions 
for i, pred in enumerate(y_pred):
    if pred == 1:
        print(f"Prediction {i + 1}: Take action A (Predicted:{pred})")
    else:
        print(f"Prediction {i + 1}: Take action B (Predicted:{pred})")
 
# Define the dataset 
x = np.array([1, 2, 3, 4, 5]) 
y = np.array([2, 4, 5, 4, 5]) 
# Define the degree of the polynomial 
degree = 2 
# Calculate the coefficients of the polynomial 
coeffs = np.polyfit(x, y, degree) 
# Print the coefficients 
print("Coefficients:", coeffs) 
# Create a polynomial function from the coefficients 
poly_func = np.poly1d(coeffs) 
# Generate x values for the polynomial curve 
x_poly = np.linspace(x.min(), x.max(), 100) 
# Generate y values for the polynomial curve 
y_poly = poly_func(x_poly) 
# Plot the data points 
plt.scatter(x, y, label="Data Points") 
# Plot the polynomial curve 
plt.plot(x_poly, y_poly, label=f"Polynomial of degree{degree}") 
# Add title and labels 
plt.title("Polynomial Regression") 
plt.xlabel("x") 
plt.ylabel("y") 
# Display the plot 
plt.legend() 
plt.show()
