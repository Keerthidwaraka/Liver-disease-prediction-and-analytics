
# Liver Disease Prediction and Analysis

Liver diseases pose a significant global health challenge, affecting millions of lives annually. This analysis aims to understand the intricate relationships between various health indicators and predict the likelihood of liver disease using the Indian Liver Patient Records dataset. Through advanced data exploration, insightful visualizations, and predictive modeling, we unravel the complexities of liver health.


## Dataset

The Indian Liver Patient Records dataset serves as our primary source of health-related information. Spanning factors such as age, gender, bilirubin levels, liver enzymes, and more, this comprehensive repository offers a rich landscape for our analysis. With a focus on predicting liver disease, we delve into patterns, correlations, and potential predictors hidden within this dataset.

## Who would benefit?
- Healthcare Professionals
- Public Health Authorities
- Data Scientists and Researchers
- Patients and General Public
## Methodology

1. Data Loading and Preprocessing: Ensuring data integrity and preparing it for analysis.
2. Exploratory Data Analysis: Uncovering insights and patterns through visualization.
3. Model Training: Developing predictive models using logistic regression.
4. Model Evaluation: Assessing the performance and reliability of our models.
## Importing Libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
```
## Load and Explore the Dataset
```
# Load the CSV file
patients = pd.read_csv(file_path)

# Explore basic data characteristics
patients.head()
patients.shape
patients.info()

# Describe gives statistical information about NUMERICAL columns in the dataset
patients.describe(include='all')

# Which features are available in the dataset?
patients.columns
```

## Data Cleaning and Exploration
```
# Check for missing values
patients.isnull().sum()

# Fill missing values 
patients=patients.fillna(0.94)
```
## Insights:
1. General Statistics: - The dataset comprises 583 entries with 11 columns. The features include 'Age,' 'Gender,' and various health indicators. 
 
2. Descriptive Statistics: - The mean age of patients is approximately 44.75 years. - The dataset is predominantly male (441 out of 583). - Significant variations are observed in health indicators, e.g., 'Total_Bilirubin,' 'Alkaline_Phosphotase,' and 'Total_Protiens.' 

3. Missing Values: - 'Albumin_and_Globulin_Ratio' has 4 missing values, potentially requiring imputation.  

## Exploratory Data Analysis

Before diving into modeling, let's perform exploratory data analysis (EDA) to gain insights into our dataset. This includes analyzing distributions, correlations, and relationships between variables. All these various data visualizations are created using seaborn and matplotlib to explore the 
distribution of patients with and without liver disease, gender distribution, age distribution, 
pair plots, and correlation heatmap. 
```
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

# Calculate and print the number of patients based on gender
patients['Gender'].value_counts().plot.bar(color='teal')

M, F = patients['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

sns.set_style('darkgrid')
plt.figure(figsize=(25,10))
patients['Age'].value_counts().plot.bar(color='red')

plt.rcParams['figure.figsize']=(10,10)
sns.pairplot(patients,hue='Gender')

sns.pairplot(patients)

f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='purple',data=patients);
plt.show()

plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='crimson')

plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Albumin'].plot.bar(color='violet')

plt.figure(figsize=(8,6))
patients.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='teal')

corr=patients.corr()
plt.figure(figsize=(20,10)) 
sns.heatmap(corr,cmap="gray",annot=True)
```
## Insights:
1. Gender Distribution: 
- Male patients significantly outnumber female patients (441 vs. 142). 
2. Age Distribution: 
- Age distribution ranges from 4 to 90 years. 
- Liver disease seems to be more prevalent in the age group above 40. 
3. Correlation Insights: Strong correlations are observed between 
- Direct_Bilirubin & Total_Bilirubin 
- Aspartate_Aminotransferase & Alamine_Aminotransferase 
- Total_Protiens & Albumin 
- Albumin_and_Globulin_Ratio & Albumin 
4. Gender-Based Observations: 
- Protein intake appears higher in males than females. 
- Albumin levels are generally higher in males. 

## Data Preprocessing for Logistic Regression
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Check if 'Gender' is present in the columns
if 'Gender' in patients.columns:
    # One-hot encode 'Gender'
    patients = pd.get_dummies(patients, columns=['Gender'], drop_first=True)

# Define features and target variable
X = patients[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
              'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
              'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
              'Albumin_and_Globulin_Ratio', 'Gender_Male']]
y = patients['Dataset']
```

## Model Training
```
# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Build and Train the Logistic Regression Model
# Create logistic regression object
logreg = LogisticRegression()

# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

#Predict Output
log_predicted= logreg.predict(X_test)
logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)

#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
```

## Model Evaluation using Cross-Validation
```
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
logmodel = LogisticRegression(C=1, penalty='l2')

# Change penalty to 'l2'
results = cross_val_score(logmodel, X_train, y_train, cv=kfold)

# Display Cross-Validation Results
print(results)
print("Accuracy:", results.mean() * 100)
```
## Insights:
1. Model Performance: 
- The logistic regression model achieved a training score of 74.26% and a test score of 66.29%. 
- Cross-validation results suggest reasonable consistency. 
2. Prediction Insights: 
- The model identifies patterns indicative of liver disease, with a notable accuracy of approximately 73.28%. 

## Discussion: 
1. Descriptive Statistics and Visualization Insights: 
The descriptive statistics and visualizations offer valuable insights into the dataset. For example, the 
mean age of patients is approximately 44.75 years, indicating a relatively mature patient population. 
Gender distribution reveals a significant prevalence of males (441 out of 583), suggesting potential 
gender-based variations in liver disease occurrence.

- How Does it Support Public Health/Health Informatics? 
Understanding the age and gender distribution is crucial for tailoring health interventions. For 
instance, if liver disease is more prevalent in older age groups, public health campaigns can target 
this demographic for early screenings and awareness programs. Gender-based variations may guide 
the development of gender-specific health policies. 

2. Logistic Regression Model Performance: 
The logistic regression model achieved a training score of 74.26% and a test score of 66.29%. 
Cross-validation results indicate reasonable consistency, suggesting that the model has the ability to 
predict liver disease based on the selected features.

- How Does it Support Public Health/Health Informatics? 
A well-performing predictive model can aid healthcare practitioners in identifying individuals at 
risk of liver disease. This information can inform personalized healthcare plans, facilitate early 
interventions, and optimize resource allocation in healthcare settings. 

3. Prediction Insights: 
The model identifies patterns indicative of liver disease, with a notable accuracy of approximately 
73.28%. Factors such as age above 40, male gender prevalence, elevated bilirubin levels, protein 
imbalances, and increased enzyme levels are highlighted. 

- How Does it Support Public Health/Health Informatics?
These identified patterns serve as actionable insights for healthcare professionals. They can guide 
the development of targeted health interventions, risk assessment strategies, and preventive 
measures. Understanding the specific indicators associated with liver disease enhances the precision 
of public health initiatives. 

## Conclusion: 
Individuals who are more prone to liver disease based on the dataset are likely to be: 
- Above the age of 40. 
- Male, as the dataset predominantly consists of male patients. 
- Exhibiting higher levels of Bilirubin, especially Total_Bilirubin and Direct_Bilirubin. 
- Demonstrating imbalances in protein levels, with a potential impact on liver health. 
- Showing elevated levels of enzymes like Aspartate_Aminotransferase and Alamine_Aminotransferase, indicating liver stress or damage. 

It's important to note that these observations are based on the provided dataset, and the 
identification of individuals prone to liver disease may vary based on additional factors not included in the current analysis. Further exploration and refinement of the predictive model can contribute to a more accurate understanding of susceptibility factors. 