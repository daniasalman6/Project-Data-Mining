#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 



# %%
# Read data
data = pd.read_csv("PCS.csv")

#%% [markdown]
## 1. Background




#%% [markdown]
## 2. Data Cleaning
# We converted the entries labeled “UNKNOWN” into missing values and removed them to ensure that the dataset is clean and suitable for analysis.


# %%
# Data cleaing
# Transfer "UNKNOWN" to missing value
# Remove missing value

# Choose variables we need in all smart questions
cols_all = [
    "Mental Illness", "Living Situation",
    "Race", "Sex", "Age Group",
    "Education Status", "SSI Cash Assistance", "SSDI Cash Assistance",
    "Public Assistance Cash Program", "Other Cash Benefits",
    "Medicaid Insurance", "Medicare Insurance", "Private Insurance",
    "Employment Status"
]

data1 = data[[c for c in cols_all if c in data.columns]].copy()

data1 = data1.replace('UNKNOWN', np.nan)
data1.isna().sum()

data1_clean = data1.dropna()
print(data1_clean.shape)



#%% [markdown]
## 3. Descriptive Analysis

# %%
# Data description
data1_clean.info()
data1_clean.describe()

#%% [markdown]
# After cleaning the data, we have 14 variables and 146,737 observations, with no missing values.
# This gives us a clean dataset that is ready for analysis.
#%% [markdown]
## 4. Interpreting results


#%% [markdown]
### Smart question 1



#%% [markdown]
### Smart question 2



#%% [markdown]
### Smart question 3
# How effectively can employment and related socioeconomic variables predict an individual’s living situation category (independent, family-based, institutional or unstable, sheltered) in New York State using PCS 2019?

#%%
# Q3

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Select data for Q3
cols_q3 = [
    "Living Situation", 
    "Employment Status", "Age Group", "Education Status",
    "SSI Cash Assistance", "SSDI Cash Assistance",
    "Public Assistance Cash Program", "Other Cash Benefits",
    "Medicaid Insurance", "Medicare Insurance", "Private Insurance"
]
data_q3 = data1_clean[cols_q3].copy()


# transfer independent variables to dummies
X = pd.get_dummies(
    data_q3.drop("Living Situation", axis=1),
    drop_first=True
)

# Encoder independent
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(data_q3["Living Situation"])


# Split and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Setup Multinomial Logistic Regression
model_q3 = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)
model_q3.fit(X_train, y_train)
y_pred = model_q3.predict(X_test)

# result
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

#%%
# Visualization

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Multinomial Logistic Regression")
plt.show()


#%% [markdown]
# The model shows moderate performance in predicting the different Living Situation categories.
# It identifies Private Residence with high precision (0.93), but the recall is lower (0.48),
# meaning that some true Private cases are not captured.
# For Institutional and Other Living Situation, the precision is also lower,
# suggesting that the model sometimes mixes these categories.
#
# One possible explanation is the class imbalance in the dataset,
# along with features that may not strongly separate the three groups.
#
# The confusion matrix shows that the model performs best for Private Residence,
# while the other categories have more overlap, which indicates that
# some living situations share similar patterns and are harder to distinguish.


#%% [markdown]
## 5. Conclusion



#%% [markdown]
## 6. Recommendations



#%% [markdown]
## 7. References