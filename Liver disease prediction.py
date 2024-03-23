#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Install Kaggle Package
get_ipython().system('pip install kaggle')


# In[2]:


# Step 2: Set Kaggle Credentials
import os
os.environ['KAGGLE_USERNAME'] = "keerthikovelamudi"
os.environ['KAGGLE_KEY'] = "68882ea8f0478446c20392ee74fff7b0"


# In[3]:


# Step 3: List Available Datasets on Kaggle
get_ipython().system('kaggle datasets list')


# In[4]:


# Step 4: Download the Indian Liver Patient Records Dataset
get_ipython().system('kaggle datasets download -d uciml/indian-liver-patient-records')


# In[5]:


# Step 5: Extract the Dataset
import zipfile

# Specify the correct file name with the ".zip" extension
zip_file_name = 'indian-liver-patient-records.zip'

with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall('kovel')


# In[6]:


# List files and directories in the 'kovel' directory
kovel_contents = os.listdir('kovel')
print("Contents of 'kovel' directory:", kovel_contents)


# In[7]:


# Step 6: Load and Explore the Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


import os

# Specify the file path in the 'kovel' directory
file_path = 'kovel/indian_liver_patient.csv'

# Read the CSV file
patients = pd.read_csv(file_path)

# Display the first few rows of the dataframe
patients.head()


# In[9]:


patients.shape


# In[10]:


patients.info()


# In[11]:


#Describe gives statistical information about NUMERICAL columns in the dataset
patients.describe(include='all')


# In[12]:


#Which features are available in the dataset?
patients.columns


# In[13]:


# Step 7: Data Cleaning and Exploration
# Check for missing values
patients.isnull().sum()


# In[14]:


# Fill missing values in 'Albumin_and_Globulin_Ratio' with the mean
patients['Albumin_and_Globulin_Ratio'].mean()


# In[15]:


patients=patients.fillna(0.94)


# In[16]:


# Check again for missing values
patients.isnull().sum()


# In[17]:


# Step 8: Data Visualization
# Various plots using seaborn and matplotlib to explore the data
# Use sns.countplot to visualize the distribution of patients with and without liver disease
sns.countplot(data=patients, x = 'Dataset', label='Count')

# Calculate and print the number of patients diagnosed with liver disease and those not diagnosed
LD, NLD = patients['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)

# Add labels and title to the plot
plt.xlabel('Diagnosis (1: Liver Disease, 0: No Liver Disease)')
plt.ylabel('Count')
plt.title('Distribution of Liver Disease Diagnosis in the Dataset')
plt.legend()
plt.show()


# In[18]:


# Calculate and print the number of patients based on gender
patients['Gender'].value_counts().plot.bar(color='teal')

M, F = patients['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[19]:


sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
patients['Age'].value_counts().plot.bar(color='red')


# In[20]:


plt.rcParams['figure.figsize']=(10,10)
sns.pairplot(patients,hue='Gender')


# In[21]:


sns.pairplot(patients)


# In[22]:


f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='purple',data=patients);
plt.show()


# In[23]:


plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='crimson')


# In[24]:


plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Albumin'].plot.bar(color='violet')


# In[25]:


plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='teal')


# In[26]:


corr=patients.corr()


# In[27]:


plt.figure(figsize=(20,10)) 
sns.heatmap(corr,cmap="gray",annot=True)


# In[28]:


# Step 9: Data Preprocessing for Logistic Regression
from sklearn.model_selection import train_test_split


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[30]:


# Check if 'Gender' is present in the columns
if 'Gender' in patients.columns:
    # One-hot encode 'Gender'
    patients = pd.get_dummies(patients, columns=['Gender'], drop_first=True)


# In[31]:


# Define features and target variable
X = patients[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
              'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
              'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
              'Albumin_and_Globulin_Ratio', 'Gender_Male']]
y = patients['Dataset']


# In[32]:


# Step 10: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[33]:


# Step 11: Build and Train the Logistic Regression Model
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[34]:


#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)


# In[35]:


#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)


# In[36]:


# Step 12: Model Evaluation using Cross-Validation
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logmodel = LogisticRegression(C=1, penalty='l2')  # Change penalty to 'l2'
results = cross_val_score(logmodel, X_train, y_train, cv=kfold)

# Display Cross-Validation Results
print(results)
print("Accuracy:", results.mean() * 100)

