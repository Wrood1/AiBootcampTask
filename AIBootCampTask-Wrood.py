import numpy as np  # Importing NumPy for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
import pandas as pd  # Importing Pandas for data manipulation
import seaborn as sns  # Importing Seaborn for enhanced visualizations
from sklearn.ensemble import GradientBoostingClassifier  # Importing GradientBoostingClassifier for building the classification model

# Reading the CSV file containing the dataset
dataset = pd.read_csv('dataset.csv')
print(dataset.head())  # Printing the first few rows of the dataset to explore
print(dataset.shape)  # Displaying the dimensions of the dataset (number of rows and columns)
print(dataset.info())  # Displaying a concise summary of the dataset, including column names, non-null counts, and data types
print(dataset.describe())  # Generating descriptive statistics for the dataset, such as mean, median, and standard deviation

# Dropping unnecessary columns
dataset = dataset.drop(['UTC', 'CNT'], axis=1)

# Splitting the dataset into features (X) and target (y)
X = dataset.drop('Fire Alarm', axis=1)  # Features (all columns except the target)
y = dataset['Fire Alarm']  # Target (the column we want to predict)

print(dataset.info())  # Displaying a concise summary of the dataset, including column names, non-null counts, and data types

from sklearn.model_selection import train_test_split  # Importing train_test_split to split the dataset into training and testing sets

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# random_state=0 ensures reproducibility of the results
# test_size=0.2 means 20% of the data will be used for testing and 80% for training

# Printing the dimensions of the training and testing sets to verify the split
print("Training feature matrix shape:", X_train.shape)
print("Testing feature matrix shape:", X_test.shape)
print("Training target vector shape:", y_train.shape)
print("Testing target vector shape:", y_test.shape)

# Creating and training the GradientBoostingClassifier model
classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the target values for the testing set
y_pred = classifier.predict(X_test)

# Printing the value counts of the target variable
print(y.value_counts())

# Evaluating the model using common classification metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)  # Accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
class_report = classification_report(y_test, y_pred)  # Classification report

# Printing the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plotting actual vs predicted values using scatter plot to visualize the model performance
plt.figure(figsize=(14, 8))  # Creating a new figure with specified size

# Define colors
actual_color = '#FF6347'  # Color for actual data points (Tomato)
predicted_color = '#4682B4'  # Color for predicted regression line (SteelBlue)

# Plot actual data points
plt.scatter(range(len(y_test)), y_test, color=actual_color, label='Actual Fire Alarm', edgecolor='black', s=100)  
# Scatter plot for actual fire alarm data
# edgecolor='black' adds a black edge around each point
# s=100 sets the size of the points

# Plot predicted data points
plt.scatter(range(len(y_test)), y_pred, color=predicted_color, label='Predicted Fire Alarm', alpha=0.6)
# Scatter plot for predicted fire alarm data
# alpha=0.6 adds transparency to the points

# Adding titles and labels with improved font sizes and styles
plt.title('Actual vs Predicted Fire Alarm (Testing set)', fontsize=22, fontweight='bold', color='#333333')  
plt.xlabel('Sample index', fontsize=16, color='#333333')  
plt.ylabel('Fire Alarm', fontsize=16, color='#333333')  

# Adding a grid with custom styling
plt.grid(True, linestyle='--', alpha=0.6, color='#999999')  

# Adding a legend with an improved font size
plt.legend(fontsize=14, loc='best', fancybox=True, framealpha=0.7, shadow=True, borderpad=1)  

# Adding a background style
plt.gca().set_facecolor('#f5f5f5')  # Setting the background color of the plot area to light gray

# Display the plot
plt.show()  

# Plotting the distribution of prediction errors
errors = y_test.values - y_pred
plt.figure(figsize=(14, 8))
sns.histplot(errors, bins=20, kde=True)
plt.title('Distribution of Prediction Errors', fontsize=22, fontweight='bold', color='#333333')
plt.xlabel('Error', fontsize=16, color='#333333')
plt.ylabel('Frequency', fontsize=16, color='#333333')
plt.grid(True, linestyle='--', alpha=0.6, color='#999999')
plt.gca().set_facecolor('#f5f5f5')
plt.show()

# Plotting prediction errors over the samples
plt.figure(figsize=(14, 8))
plt.scatter(range(len(errors)), errors, alpha=0.6, color='b')
plt.title('Prediction Errors', fontsize=22, fontweight='bold', color='#333333')
plt.xlabel('Sample index', fontsize=16, color='#333333')
plt.ylabel('Error', fontsize=16, color='#333333')
plt.grid(True, linestyle='--', alpha=0.6, color='#999999')
plt.gca().set_facecolor('#f5f5f5')
plt.show()
