# **Phishing Website Detection**

**Repository Description**: Detect phishing websites using Logistic Regression on UCI dataset features. Built in Jupyter Notebook with 73% accuracy. Explore data prep and model evaluation!

## **Table of Contents**

- Project Overview
- Dataset
- Methodology
- Model Performance
- Setup Instructions
- File Structure
- Future Improvements
- License

## **Project Overview**

This project develops a machine learning model to detect phishing websites using URL-based features from the UCI Phishing Websites Dataset. A Logistic Regression model is trained in a Jupyter Notebook environment using Python, with libraries like pandas, scikit-learn, seaborn, and matplotlib. The project focuses on preprocessing the dataset, handling duplicates, training, evaluating, and saving the model.

**Key Objectives**:

- Preprocess the UCI dataset to remove duplicates.
- Train a Logistic Regression model with 7 URL-based features.
- Evaluate performance with accuracy, precision, recall, and a confusion matrix.
- Save the trained model as a .pkl file.

## **Dataset**

- **Source**: <https://archive.ics.uci.edu/ml/datasets/Phishing+Websites>
- **File**: Training Dataset.arff
- **Rows**: 11,055 (5,206 duplicates removed, 5,849 unique rows)
- **Columns**: 31 (30 features + 1 label: Result)
- **Label**: Result (-1 for phishing, 1 for legitimate, mapped to 0 and 1)
- **Features Used**:
  - having_IP_Address
  - URL_Length
  - having_At_Symbol
  - double_slash_redirecting
  - Prefix_Suffix
  - having_Sub_Domain
  - HTTPS_token

**Label Distribution**

- Legitimate (1):
- Phishing (-1):

## **Methodology**

The project is implemented in scripts/load_dataset.ipynb. The following code performs the workflow, with each phase described below.

Import Libraries  
-Load required Python libraries.

Load the UCI Dataset  
-Read Training Dataset.arff into a pandas DataFrame.

Decode Byte Strings  
-Convert byte-string columns to strings.

Inspect the Dataset  
-Display dataset structure and labels.

Clean the Dataset  
-Remove duplicates.

Select Features and Labels  
-Extract features and labels.

Convert Data Types  
-Convert to numeric types.

Split Data  
-Split into training and test sets.

Train the Model  
-Train Logistic Regression.

Evaluate the Model  
-Compute metrics and plot the confusion matrix.

Save the Model  
-Save the trained model.

## Import libraries
``` python
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
print("Libraries imported successfully")
```
Libraries imported successfully


## Load ARFF file
```python
arff_file = 'data/Training Dataset.arff'
try:
    data_arff, meta = arff.loadarff(arff_file)
    df = pd.DataFrame(data_arff)
    print("Dataset loaded successfully")
except Exception as e:
    print("Error loading ARFF:", e)
    raise
```
Dataset loaded successfully


## Decode byte strings
```python
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')
print("Byte strings decoded")
```
Byte strings decoded


## Inspect data
```python
print("Raw data head:")
print(df.head())
print("\nRaw data shape:", df.shape)
print("Raw data columns:", df.columns.tolist())
print("\nLabel column (Result) unique values:")
print(df['Result'].value_counts())
```
Raw data head:
  having_IP_Address URL_Length Shortining_Service having_At_Symbol  \
0                -1          1                  1                1   
1                 1          1                  1                1   
2                 1          0                  1                1   
3                 1          0                  1                1   
4                 1          0                 -1                1   

  double_slash_redirecting Prefix_Suffix having_Sub_Domain SSLfinal_State  \
0                       -1            -1                -1             -1   
1                        1            -1                 0              1   
2                        1            -1                -1             -1   
3                        1            -1                -1             -1   
4                        1            -1                 1              1   

  Domain_registeration_length Favicon  ... popUpWidnow Iframe age_of_domain  \
0                          -1       1  ...           1      1            -1   
1                          -1       1  ...           1      1            -1   
2                          -1       1  ...           1      1             1   
3                           1       1  ...           1      1            -1   
4                          -1       1  ...          -1      1            -1   

  DNSRecord web_traffic Page_Rank Google_Index Links_pointing_to_page  \
0        -1          -1        -1            1                      1   
1        -1           0        -1            1                      1   
2        -1           1        -1            1                      0   
3        -1           1        -1            1                     -1   
4        -1           0        -1            1                      1   

  Statistical_report Result  
0                 -1     -1  
1                  1     -1  
2                 -1     -1  
3                  1     -1  
4                  1      1  

[5 rows x 31 columns]

Raw data shape: (11055, 31)
Raw data columns: ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report', 'Result']

Label column (Result) unique values:
Result
1     6157
-1    4898
Name: count, dtype: int64


## Clean dataset
```python
df = df.drop_duplicates()
print("Rows after removing duplicates:", len(df))
```
Rows after removing duplicates: 5849


## Select features and labels
```python
features = [
    'having_IP_Address', 'URL_Length', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain',
    'HTTPS_token'
]
# Optional: Use all features
# features = [col for col in df.columns if col != 'Result']
X = df[features]
y = df['Result']
print("Features selected:", features)
print("X shape:", X.shape)
print("y shape:", y.shape)
```
Features selected: ['having_IP_Address', 'URL_Length', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'HTTPS_token']
X shape: (5849, 7)
y shape: (5849,)


## Convert data types
```python
try:
    X = X.astype(float).astype(int)
    y = y.map({'-1': 0, '1': 1, -1: 0, 1: 1, -1.0: 0, 1.0: 1})
    print("Data types converted")
    print("X dtypes:", X.dtypes)
    print("y dtype:", y.dtype)
except ValueError as e:
    print("Conversion error:", e)
    for col in X.columns:
        print(f"{col} unique values:", X[col].unique())
    raise
```
Data types converted
X dtypes: having_IP_Address           int64
URL_Length                  int64
having_At_Symbol            int64
double_slash_redirecting    int64
Prefix_Suffix               int64
having_Sub_Domain           int64
HTTPS_token                 int64
dtype: object
y dtype: int64

## Split data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```
X_train shape: (4679, 7)
X_test shape: (1170, 7)
y_train shape: (4679,)
y_test shape: (1170,)


## Train model
```python
model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
print("Model trained successfully")
```
Model trained successfully

## Evaluate model
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate']))

# Plot confusion matrix
os.makedirs('figures', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Legitimate'], yticklabels=['Phishing', 'Legitimate'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('figures/confusion_matrix.png')
plt.show()
print("Confusion matrix saved to figures/confusion_matrix.png")
```
Accuracy: 0.7307692307692307

Classification Report:
              precision    recall  f1-score   support

    Phishing       0.73      0.79      0.76       620
  Legitimate       0.74      0.67      0.70       550

    accuracy                           0.73      1170
   macro avg       0.73      0.73      0.73      1170
weighted avg       0.73      0.73      0.73      1170


## Save model
```python
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_regression_model.pkl')
print("Model saved to models/logistic_regression_model.pkl")
```
Confusion matrix saved to figures/confusion_matrix.png


## **Model Performance**

- **Algorithm**: Logistic Regression
- **Features**: 7 URL-based
- **Accuracy**: 73.08%

**Confusion Matrix**:

 

