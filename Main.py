import math
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore  # Correct import for zscore
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score

# Loading the dataset
df = pd.read_csv('dataset.csv')
df = df.rename(columns={'Nacionality': 'Nationality', 'Target': 'Student_status'})

print(df.isnull().sum())

#dropping unecessary features
df.drop('Gender', axis=1, inplace=True)
df.drop('Age at enrollment', axis=1, inplace=True)
df.drop('Displaced', axis=1, inplace=True)
df.drop('Educational special needs', axis=1, inplace=True)

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
z_scores = np.abs(zscore(df[numerical_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)] #removing outliers

print(z_scores)
print(f"Original number of rows: {df.shape[0]}")
print(f"Number of rows after removing outliers: {df_no_outliers.shape[0]}")

correlation_matrix = df_no_outliers[numerical_cols].corr()

plt.figure(figsize=(25, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap of Student Performance Data')
plt.show() #heatmap

#box plot
# Plotting the box plot for each numerical column before and after outlier removal
plt.figure(figsize=(50, 35))

for i, column in enumerate(numerical_cols, 1):
    plt.subplot(5, math.ceil(len(numerical_cols) / 5), i)

    # Box plot for original data with outliers (Red)
    sns.boxplot(data=df[column], color='red', showfliers=True)

    # Box plot for data after outlier removal (Blue)
    sns.boxplot(data=df_no_outliers[column], color='blue', showfliers=False)

    plt.title(f'Box Plot of {column}')
    plt.legend(labels=['Original with Outliers', 'After Outlier Removal'], loc='upper right')

plt.tight_layout()
plt.show()

#normalisation
columns_to_normalize = [
    'Application mode',
    'Application order',
    'Daytime/evening attendance',
    'Mother\'s qualification',
    'Father\'s qualification',
    'Mother\'s occupation',
    'Father\'s occupation',
    'Unemployment rate',
    'Inflation rate',
    'GDP',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (without evaluations)'
]
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
print(df.head())
print(df.describe())
print(df.shape[0])

#encoding using label encoder
label_encoder = LabelEncoder()
df['Student_status'] = label_encoder.fit_transform(df['Student_status'])
print(df[['Student_status']].head())

#train
X = df.drop('Student_status', axis=1)
y = df['Student_status']

# Perform an 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(random_state=42)

# Train the model on the training data
random_forest.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = random_forest.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')

# Generate a classification report
report_rf = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)
print('Random Forest Classification Report:')
print(report_rf)

# Confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print('Random Forest Confusion Matrix:')
print(conf_matrix_rf)

# Performing k-fold cross-validation
k = 5
cv_scores = cross_val_score(random_forest, X, y, cv=k)
print(f'{k}-Fold Cross-Validation Accuracy: {cv_scores.mean():.2f}')


# Feature importance
feature_importance_rf = random_forest.feature_importances_
importance_df_rf = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance_rf})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)

print('Random Forest Feature Importance:')
print(importance_df_rf)

# Assuming 'importance_df_rf' is your DataFrame with 'Feature' and 'Importance' columns
# Sort features by importance for better visualization
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=True)

# Plotting the Feature Importance
plt.figure(figsize=(30, 28))
plt.barh(importance_df_rf['Feature'], importance_df_rf['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance Bar Plot')
plt.show()

#class distribution graph
# Combine train and test distributions for the plot
y_train_dist = pd.Series(y_train).value_counts().sort_index()
y_test_dist = pd.Series(y_test).value_counts().sort_index()

# Plotting the class distribution for train and test sets
plt.figure(figsize=(10, 6))
plt.bar(y_train_dist.index - 0.2, y_train_dist.values, width=0.4, label='Train', color='blue')
plt.bar(y_test_dist.index + 0.2, y_test_dist.values, width=0.4, label='Test', color='orange')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in Train and Test Sets')
plt.xticks(y_train_dist.index, labels=label_encoder.classes_)
plt.legend()
plt.show()

#distribution plots
num_numerical_cols = len(numerical_cols)

# Calculate the number of rows and columns needed for the subplots
n_cols = 5
n_rows = math.ceil(num_numerical_cols / n_cols)  # Calculate the required number of rows

# Plot distribution plots for each numerical column
plt.figure(figsize=(20, 5 * n_rows))
for i, column in enumerate(numerical_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(df_no_outliers[column], kde=True)
    plt.title(f'Distribution Plot of {column}')
plt.tight_layout()
plt.show()

#confusion matrix
y_test_binary = [1 if label == 0 else 0 for label in y_test]
y_pred_rf_binary = [1 if label == 0 else 0 for label in y_pred_rf]

# Compute confusion matrix
conf_matrix_binary = confusion_matrix(y_test_binary, y_pred_rf_binary)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Dropout', 'Dropout'], yticklabels=['Not Dropout', 'Dropout'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Dropout vs. Not Dropout')
plt.show()