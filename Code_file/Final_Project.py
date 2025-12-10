#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_columns', None)
#%%[markdown]
## Dataset and Git Hub Link
#
# [Dataset](https://data.ny.gov/Human-Services/Patient-Characteristics-Survey-PCS-2019/urn3-ezfe/about_data)
#
# [Git Hub](https://github.com/daniasalman6/Project-Data-Mining)

# %%
# Read data
data = pd.read_csv("PCS_data")

#%% [markdown]
## 1. Background

# Mental health and housing stability are ongoing concerns in New York State. 
# Many adults face financial stress, limited insurance coverage, and varying levels of social support, 
# all of which may influence both mental-health outcomes and where a person lives. 
# Prior work shows that social determinants such as economic stability and housing conditions are closely linked to mental-health risk (Maqbool et al., 2015).
#
# The 2019 PCS dataset provides detailed information on mental illness, socioeconomic conditions, 
# and living arrangements. Because it includes both health-related and financial variables, 
# it allows us to examine how factors such as employment, education, assistance programs, 
# and insurance relate to mental-health reporting and housing categories. Studies have found that access to insurance and financial resources plays an important role in shaping mental-health outcomes (Tanarsuwongkul et al., 2025).
#
# In this project, we use the PCS data to describe these relationships and evaluate how well socioeconomic variables can predict mental illness and different types of living situations. 
# Understanding these links is important because housing and mental health often influence one another in meaningful ways (Healthy People 2030).
#%% [markdown]
## 2. Data Cleaning
# We converted the entries labeled “UNKNOWN” into missing values and removed them to ensure that the dataset is clean and suitable for analysis.

# %%
# Data cleaing
# Transfer "UNKNOWN" to missing value
# Remove missing value

# Choose variables we need in all smart questions
cols_all = [
    "Mental Illness", "Serious Mental Illness", "Living Situation",
    "Race", "Sex", "Age Group",
    "Education Status", "SSI Cash Assistance", "SSDI Cash Assistance",
    "Public Assistance Cash Program", "Other Cash Benefits",
    "Medicaid Insurance", "Medicare Insurance", "Private Insurance",
    "Employment Status", "Veteran Status", "Region Served", "Sexual Orientation", 
    "Household Composition", "Alcohol Related Disorder", 
    "Drug Substance Disorder"
]

data1 = data[[c for c in cols_all if c in data.columns]].copy()

data1 = data1.replace('UNKNOWN', np.nan)
data1.isna().sum()

data1_clean = data1.dropna()
print(f"The shape of the data is:  {data1_clean.shape}")

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

# Only keep YES/NO rows - no "Unknown"
data1_clean = data1_clean[data1_clean['Mental Illness'].isin(['NO','YES'])]
data1_clean = data1_clean[data1_clean['Serious Mental Illness'].isin(['NO','YES'])]

# Code variable of interest as 0="NO" and 1="YES"
data1_clean['Mental Illness'] = data1_clean['Mental Illness'].map({'NO': 0, 'YES': 1})
data1_clean['Serious Mental Illness'] = data1_clean['Serious Mental Illness'].map({'NO': 0, 'YES': 1})


#%%
# Create values for independent variables
categorical_cols = [
    'Race', 'Sex', 'Age Group', 'Education Status', 'Veteran Status',
    'Region Served', 'Sexual Orientation', 'Household Composition',
    'Alcohol Related Disorder', 'Drug Substance Disorder'
]

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data1_clean[col] = le.fit_transform(data1_clean[col])
    le_dict[col] = le

#%%
# Plot prevalence of Mental Illness
data1_clean['Mental Illness'] = data1_clean['Mental Illness'].map({0: 'NO', 1: 'YES'})

data1_clean['Mental Illness'].value_counts()
data1_clean['Mental Illness'].value_counts(normalize=True).plot(kind='bar', color='steelblue')
plt.title('Prevalence of Mental Illness')
plt.ylabel('Proportion')
plt.xlabel('Mental Illness')
plt.xticks(rotation=0)
plt.show()

#%%
data1_clean['Mental Illness'] = data1_clean['Mental Illness'].map({'NO': 0, 'YES': 1})

#%%[markdown]
# Over 90% of individuals in New York said they have a mental illness.

#%%
# Plot prevalence of Serious Mental Illness
data1_clean['Serious Mental Illness'] = data1_clean['Serious Mental Illness'].map({0: 'NO', 1: 'YES'})

data1_clean['Serious Mental Illness'].value_counts()
data1_clean['Serious Mental Illness'].value_counts(normalize=True).plot(kind='bar', color='steelblue')
plt.title('Prevalence of Serious Mental Illness')
plt.ylabel('Proportion')
plt.xlabel('Mental Illness')
plt.xticks(rotation=0)
plt.show()
#%%
data1_clean['Serious Mental Illness'] = data1_clean['Serious Mental Illness'].map({'NO': 0, 'YES': 1}

#%% [markdown]
# The amount of patients with a mental illness vs. a serious mental illness does not differ.
#%%[markdown]
#### Plotting all variables and their relationship with mental illness presence

#%%
def bar_plot(col):
    plot_df = data1_clean.groupby(col)['Mental Illness'].mean().reset_index()

    mapping = {code: label for code, label in enumerate(le_dict[col].classes_)}
    plot_df[f'{col}_Label'] = plot_df[col].map(mapping)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x=f'{col}_Label', y='Mental Illness')
    plt.title(f'Mental Illness Prevalence by {col}')
    plt.ylabel('Share with Mental Illness')
    plt.xlabel(col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
# Generate plots
for var in categorical_cols:
    bar_plot(var)



#%%[markdown]
# Mental illness prevalence is roughly the same among all races and genders. Adults are more
# likely to have a mental illness than children, and the more eduacted an individual is, the more 
# likely they are to have a mental illness. Suprisingly, mental health prescence does not differ 
# between those that are and are not veterans, and mental health prescence is about the same among 
# regions in New York. There is also a small difference of thos who have mental illnesses among
# straight and non-heterosexual orientations. Those who live with others have a slightly less share
# of people who have a mental illness than those who live alone. Suprinsignly, the prescence of those 
# who have a mental illness is the same among those who do and do not have alcohol related and drug 
# substance disorders. 


#%%
# defining independent and dependent variables
X = data1_clean[categorical_cols]
y = data1_clean['Mental Illness']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

#%%[markdown]
#### Logistic Regression

#%%
logit_model = LogisticRegression(max_iter=2000)
logit_model.fit(X_train, y_train)

y_pred_logit = logit_model.predict(X_test)
y_pred_prob_logit = logit_model.predict_proba(X_test)[:, 1]

print("Classification Report")
print(classification_report(y_test, y_pred_logit))

auc_logit = roc_auc_score(y_test, y_pred_prob_logit)
print("AUC:", auc_logit)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob_logit)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%[markdown]
# The logistic regression highlights the class imbalance of the data. The accuracy of the model
# appears to be 97%. However, the model does not identify those who do not have a mental illness.
# The AUC of 0.67 demonstrates that the logistic regression had modest discriminative ability. 
# The current model is not suitable for the data due to the class imbalance. 

#%%
# Logistic Regression 
logit_model = LogisticRegression(max_iter=2000)
logit_model.fit(X_train, y_train)

y_pred_logit = logit_model.predict(X_test)
y_pred_prob_logit = logit_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred_logit))

auc_logit = roc_auc_score(y_test, y_pred_prob_logit)
print("Logistic Regression AUC:", auc_logit)

# Odds ratios
odds_ratios = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logit_model.coef_[0],
    "Odds Ratio": np.exp(logit_model.coef_[0])
}).sort_values(by="Odds Ratio", ascending=False)

print("Logistic Regression Odds Ratios")
print(odds_ratios)

# Graph
plt.figure(figsize=(8, 6))
sns.barplot(data=odds_ratios, x="Odds Ratio", y="Feature")
plt.title("Odds Ratios from Logistic Regression")
plt.axvline(1, color='red', linestyle='--')  # 1 = no effect
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_logit)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_logit:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%[markdown]
# The independent/demographic variables do not explain sufficient variation in mental illness 
# prevalence. This model does not predict those without mental illness since there is significant 
# class imbalance.  

#%% [markdown]
#### Logistic Regression 2 - Accounting for  Class Imbalance
#### Balance class weight
#%%
logit_model = LogisticRegression(max_iter=1000, class_weight='balanced')
logit_model.fit(X_train, y_train)

y_pred = logit_model.predict(X_test)
y_pred_prob = logit_model.predict_proba(X_test)[:,1]

# Classification report
print("Logistic Regression Classification Report")
print(classification_report(y_test, y_pred))

# ROC AUC
auc = roc_auc_score(y_test, y_pred_prob)
print("Logistic Regression AUC:", auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC={auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%[markdown]
# When class imbalance is accounted for, a decent amount of individuals who do not have a mental 
# illness are identified. The model accuracy is now 76%, but identifies those without a mental illness. 
# The demographic/independent variables have modest predictive power now. 

#%%  
# Odds Ratios
coefficients = logit_model.coef_[0]
odds_ratios = np.exp(coefficients)

odds_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
}).sort_values(by='Odds Ratio', ascending=False)

print("Logistic Regression Odds Ratios")
print(odds_df)


# Plot odds ratios
plt.figure(figsize=(8,6))
sns.barplot(x='Odds Ratio', y='Feature', data=odds_df.sort_values('Odds Ratio', ascending=False))
plt.title('Logistic Regression Odds Ratios')
plt.show()

#%%[markdown]
# The improved logistic regression suggests that being older decreases the chance of having a mental illness, cohabitating
# with others and substance use minimally increases the risk of having a mental illness. All demographic variables have 
# small impacts on mental illnesses. 

#%%[markdown]
#### Random Forest

#%%
rf_model = RandomForestClassifier(n_estimators=500, random_state=123)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Classification Report")
print(classification_report(y_test, y_pred_rf))

auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
print("AUC:", auc_rf)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%[markdown]
# The Random Forest model also had an accuracy of 97% and an AUC of 0.69, again having some 
# discriminative ability. This model also suffers from the class imbalance, having a recall of 0.04
# for individuals without a mental illness. This model is extremely biased toward those who do have
# a mental illness.
#%%
#### Random Forest Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

print(feat_imp_df)

sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance - Random Forest')
plt.show()

#%%[markdown]

# The random forest model demonstrates that region served, education status, and race are the most 
# significant predictors of mental illness. Sexual orientation, household composition, and age group 
# have some importance on the model. Sex, alcohol related disorder, drug substance disorder, and 
# veteran status do not contribute much to the model's predictions. 
#%%[markdown]
#### Fit Random Forest with balanced class weighting

#%%
rf_model = RandomForestClassifier(n_estimators=500, random_state=123, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:,1]

# Classification report
print("Classification Report")
print(classification_report(y_test, y_pred))

# ROC AUC
auc = roc_auc_score(y_test, y_pred_prob)
print("AUC:", auc)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'Random Forest (AUC={auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.show()

#%%[markdown]

# When the class is balanced, the random forest model had a 75% accuracy and an AUC of 0.68.
# The recall for those without a mental illness is 0.55 which improved from 0.04. 
#%%[markdown]
#### Feature importances

#%%
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("Feature Importances")
print(feat_df)

# Plot feature importances
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

#%%[markdown]
# Region served, education status, and age group are the most influential predictors
# of mental illness. Race, sexual orientation, and household consumption have some importance
# on predicting mental illness. Sex, alchohol related disorder, drug substance disorder, and
# veteran status did not contribute much to the model.

#%%[markdown]
#### Correlation heatmap

#%%
plt.figure(figsize=(12,8))
sns.heatmap(data1_clean[categorical_cols + ['Mental Illness']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#%%[markdown]
# No correlation is above 0.51, so I do not think multicollinearity needs to be addressed.
# %%



#%% [markdown]
### Smart question 2
# To what extent can socioeconomic factors, particularly educational attainment, predict mental health diagnosis among adults in New York State using PCS 2019, and which variables are the strongest predictors

#%%
data.head()

#%%[markdown]
# After manually looking at the dataset, I have shortlisted the following variables (socioeconomic factors) from the dataset that might impact mental health diagnosis
#### Filter by: 
# - Age group - "ADULT"
#### Independent variables/features:
# - Living situation: ‘Private Residence’, ‘Institutional Setting’, ’Other Living Situation’, or ‘Unknown’
# - Household compostion: ‘Lives Alone’, ’Cohabitates with Others’, ‘Not Applicable’, or ‘Unknown’.
# - Employment status: ‘Employed’, ‘Non-paid/Volunteer’, ‘Not In Labor Force: Unemployed and not looking for work’, ‘Unemployed, looking for work’, or ‘Unknown Employment Status’.
# - Number of hours worked each week: ‘01-14 Hours’, ‘15-34’, ‘35 Hours or More’, ‘Not Applicable’, or ‘Unknown Employment Hours’
# - Education Status: ‘No Formal Education’, ‘Pre-K to Fifth Grade’, ‘Middle School to High School’, ‘Some College’, ‘College or Graduate Degree’, ‘Other’, or ‘Unknown’.
# - Special Education services: ‘Yes’, ‘No’, ‘Not Applicable’, ‘Unknown’.
# - Mental Illness: ‘Yes’, ‘No’, or ‘Unknown’.
# - SSI Cash Assistance: ‘Yes’, ‘No’, or ‘Unknown’
# - SSDI Cash Assistance: ‘Yes’, ‘No’, or ‘Unknown’
# - Veterans Disability Benefits: ‘Yes’, ‘No’, or ‘Unknown’
# - Veterans Cash Assistance: ‘Yes’, ‘No’, or ‘Unknown’
# - Public Assistance Cash Program: ‘Yes’, ‘No’, or ‘Unknown’
# - Other Cash Benefits: ‘Yes’, ‘No’, or ‘Unknown’
# - Medicaid and Medicare Insurance: ‘Yes’, ‘No’, or ‘Unknown’
# - No Insurance: ‘Yes’ – Indicates individual DOES NOT have any health insurance; ‘No’ – Indicates individual has at least one type of health insurance; ‘Unknown’ – Indicates that it is not known whether individual has health insurance
# - Unknown Insurance Coverage: ‘Yes’ indicates that it is not known whether individual has health insurance; ‘No’ indicates individual has at least one type of health insurance
# - Medicaid Insurance: ‘Yes’, ‘No’, or ‘Unknown’
# - Medicaid Managed Insurance: ‘Yes’, ‘No’, ‘Not Applicable’, ‘Unknown’.
# - Private Insurance: ‘Yes’, ‘No’, or ‘Unknown’
# - Child Health Plus Insurance: ‘Yes’, ‘No’, or ‘Unknown’
# - Other Insurance: ‘Yes’, ‘No’, or ‘Unknown’
# - Criminal Justice status: ‘Yes’, ‘No’, or ‘Unknown’
#### Target Variable
# - Mental Illness: ‘Yes’, ‘No’, or ‘Unknown’

#### Data Cleaning and EDA for Q2

#%%[markdown]
#### Subset Data

#%%
""""Subsetting data to filter out all infromation for people who lie in the "ADULT" age group"""
q2_df = data[data["Age Group"] == 'ADULT']

"""filter all required rows shortlisted above"""
q2_cols = ["Living Situation", "Household Composition", "Employment Status", "Number Of Hours Worked Each Week", "Education Status", "Special Education Services", "Mental Illness", "SSI Cash Assistance", "Veterans Disability Benefits", "Veterans Cash Assistance", "Public Assistance Cash Program", "Other Cash Benefits", "Medicaid and Medicare Insurance", "No Insurance", "Unknown Insurance Coverage", "Medicaid Insurance", "Medicaid Managed Insurance", "Private Insurance", "Child Health Plus Insurance", "Other Insurance", "Criminal Justice Status"]

q2_df = q2_df[[c for c in q2_cols]]

#%%[markdown]
#### Removing UNKNOWN category from Mental Illness variable since it is not meaningful as a class label

#%%
q2_df = q2_df[q2_df["Mental Illness"].isin(["YES", "NO"])]

#%%[markdown]
#### Inspect distribution for target variable "Mental Illness"

#%%
print(q2_df["Mental Illness"].value_counts(dropna=False))
print(q2_df["Mental Illness"].value_counts(normalize=True) * 100)

#%%[markdown]
#### EDA for Q2
#### The first step is to look for distribution of each contesting predictor

#%%
def check_dist(df: pd.DataFrame):
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(q2_df[col].value_counts(dropna=False))

check_dist(q2_df)
#%%[markdown]
#### Dropping variables that are not useful by looking at their ditribution

#%%
"""special education is 99% "NOT APPLICABLE" which means no predictive power so let's drop it"""
q2_df = q2_df.drop(columns=["Special Education Services"])

#%%[markdown]
#### Collapsing or merging rare or small categories

#%%
"""--- Education Status: merge rare levels ---"""
q2_df["Education Status"] = q2_df["Education Status"].replace({
    "NO FORMAL EDUCATION": "LOW EDUC",
    "PRE-K TO FIFTH GRADE": "LOW EDUC"
})

"""--- Hours Worked: simplify to WORKING vs NOT WORKING ---"""
q2_df["Working Status"] = np.where(
    q2_df["Number Of Hours Worked Each Week"] == "NOT APPLICABLE",
    "NOT WORKING",
    "WORKING"
)

# drop original detailed hours
q2_df = q2_df.drop(columns=["Number Of Hours Worked Each Week"])

# --- Veterans Cash Assistance: merge very rare category ---
q2_df["Veterans Cash Assistance"] = q2_df["Veterans Cash Assistance"].replace({
    "YES": "ANY_ASSISTANCE",
    "NO": "NO_ASSISTANCE",
    "UNKNOWN": "NO_ASSISTANCE"
})

# --- Child Health Plus: merge rare YES into binary ---
q2_df["Child Health Plus Insurance"] = np.where(
    q2_df["Child Health Plus Insurance"] == "YES",
    "YES",
    "NO"
)

check_dist(q2_df)

#%%[markdown]
#### Run chi square tests between dependent and independent variables

#%%
predictors = [c for c in q2_df.columns if c != "Mental Illness"]

for col in predictors:
    table = pd.crosstab(q2_df[col], q2_df["Mental Illness"])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"{col}: p-value = {p}")

print("Out of all variables except Veterans Cash Assistance, the p-values are low so all are statistically significant")

#%%[markdown]
#### Dropping Veterans Cash Assistance Column

#%%
q2_df = q2_df.drop(columns=["Veterans Cash Assistance"])
predictors.remove("Veterans Cash Assistance")

#%%[markdown]
#### Check for multicollinearity now

#%%
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

vars_list = q2_df.columns  
cramer_matrix = pd.DataFrame(index=vars_list, columns=vars_list, dtype=float)

for c1 in vars_list:
    for c2 in vars_list:
        cramer_matrix.loc[c1, c2] = cramers_v(q2_df[c1], q2_df[c2])

"""Making a visual correlation heatmap"""
fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(cramer_matrix.astype(float)) 

# Add numbers
for i in range(len(vars_list)):
    for j in range(len(vars_list)):
        ax.text(j, i, f"{cramer_matrix.iloc[i, j]:.2f}",
                ha="center", va="center")

# Labels
ax.set_xticks(np.arange(len(vars_list)))
ax.set_yticks(np.arange(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=90)
ax.set_yticklabels(vars_list)

plt.title("Cramér’s V Correlation Heatmap (Categorical Variables)")
plt.tight_layout()
plt.show()

#%%[markdown]
#### Drop highly related variables

#%%
"""The highly correlated variables are:
   - Household Composition and Living Situation
   - Employment Status and Working Status
   - Medicaid And Medicare Insurance and Other Insurance
   - Medicaid And Medicare Insurance and Private Insurance
   - Medicaid And Medicare Insurance and Medicaid Insurance
   - Medicaid And Medicare Insurance and Unknown Insurance Coverage
   - Unknown Insurance Coverage and Medicaid Insurance
   - Medicaid Insurance and No Insurance
   - Medicaid Managed Insurance and Medicaid Insurance
   - Private Insurance and Other Insurance"""

drop_cols =  ["Household Composition", "Employment Status", "Other Insurance", "Private Insurance", "Medicaid Insurance", "Unknown Insurance Coverage","No Insurance"]

q2_df = q2_df.drop(columns=drop_cols)
for i in drop_cols:
    predictors.remove(i)
predictors

#%%
"""encode categorical columns"""
q2_df["Mental Illness"] = q2_df["Mental Illness"].replace({"YES":1, "NO":0}).astype(int)

q2_df = pd.get_dummies(q2_df, columns=predictors, drop_first=False)

q2_df = q2_df.astype(float)

#%%[markdown]
#### Build the regression model

#%%

X = q2_df.drop(columns=["Mental Illness"])
y = q2_df["Mental Illness"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

from sklearn.metrics import classification_report

log_model = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    solver='lbfgs'
)

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("=== Logistic Regression (Balanced) ===")
print(classification_report(y_test, log_pred))

#%%[markdown]
##### The balanced logistic regression model shows that it performs well in identifying the majority class (1.0), achieving very high precision (0.99) and a strong recall (0.63) for this class. This means when the model predicts “1”, it is almost always correct, and it captures most of the actual 1’s. However, performance on the minority class (0.0) remains weak: precision is extremely low (0.04), meaning most predicted 0’s are actually incorrect, and although recall is higher (0.72), the very low precision results in a poor F1-score (0.07). Overall accuracy is 0.63, but this is driven almost entirely by the model’s success on the majority class. These results indicate that even with class balancing, logistic regression struggles to meaningfully learn the minority “0” class due to severe class imbalance and limited separation in the features.

#%%

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("=== Random Forest (Class-Weighted) ===")
print(classification_report(y_test, rf_pred))


#%%[markdown]
##### The class-weighted Random Forest shows a clear improvement over logistic regression, especially in identifying the majority class (1.0). Precision for class 1 remains extremely high (0.99), and recall increases to 0.72, leading to a strong F1-score of 0.83. Performance on the minority class (0.0) is still limited due to the very small support, while recall from changes from 0.72 (logistic) to 0.58, while precision remains low (0.04). Overall accuracy rises to 0.72, reflecting better learning of nonlinear patterns in the data. Although the minority class is still challenging, Random Forest provides a more robust and balanced performance than logistic regression under severe class imbalance.

#%%
# Get raw feature importances
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Group by original variable name (everything before the first underscore)
feat_df['OriginalFeature'] = feat_df['Feature'].str.split('_').str[0]

# Sum importances for all dummy columns from the same variable
grouped_importance = (
    feat_df.groupby('OriginalFeature')['Importance']
           .sum()
           .sort_values(ascending=False)
)

print(grouped_importance)

plt.figure(figsize=(10,6))
sns.barplot(
    x=grouped_importance.values,
    y=grouped_importance.index
)
plt.title("Feature Importance by Original Variable")
plt.xlabel("Importance")
plt.ylabel("Original Feature")
plt.tight_layout()
plt.show()
#%%[markdown]
### The Random Forest shows that several socioeconomic factors contribute to predicting mental health diagnoses. Education status is the strongest overall predictor, followed closely by Medicaid-related insurance and SSI cash assistance, indicating that financial and support-related factors matter most. Criminal justice status, living situation, and working status also provide moderate signal, while other benefit programs and child health insurance contribute smaller effects. Overall, the model uses many factors, but each provides only modest predictive power.


#%%[markdown]
#### Conclusion for Q2:
#### Overall, predicting mental illness from the available factors is difficult because the data is highly imbalanced. Logistic regression struggles with the minority class, while the Random Forest performs better but still cannot reliably identify class 0. Feature importance shows that education, Medicaid insurance, and SSI assistance are the strongest predictors, with other socioeconomic factors contributing smaller effects. In general, the models capture some patterns, but the features provide only limited power for accurate classification.
#%% [markdown]
### Smart question 3
# How effectively can employment and related socioeconomic variables predict an individual’s living situation category (independent, family-based, institutional or unstable, sheltered) in New York State using PCS 2019?



#%%
# Select data for Q3
cols_q3 = [
    "Living Situation", 
    "Employment Status", "Age Group", "Education Status",
    "SSI Cash Assistance", "SSDI Cash Assistance",
    "Public Assistance Cash Program", "Other Cash Benefits",
    "Medicaid Insurance", "Medicare Insurance", "Private Insurance"]

data_q3 = data1_clean[cols_q3].copy()

#%%[markdown]
#### Check for the imbalance issue on Y

#%%
print(data_q3["Living Situation"].value_counts(dropna=False))
print(data_q3["Living Situation"].value_counts(normalize=True) * 100)

#%%[markdown]
# Living Situation is highly imbalanced, with Private Residence taking up most of the data
# and Institutional Setting appearing in less than 1%, which makes models lean toward the
# majority class. Later we apply class weighting and SMOTE to ease this issue.

#%%[markdown]
#### Checking distributions

#%%
for col in data_q3.columns:
    if col == "Living Situation":
        continue
    print("\n", data_q3[col].value_counts(dropna=False))

#%%[markdown]
# Most predictors show reasonable variation, but a few categories are much smaller than the rest. For example, 
# “Non-paid/Volunteer” in Employment Status and “No Formal Education” in Education Status have very small counts compared to other groups, 
# while the assistance and insurance variables are skewed but still have enough cases to be useful for modeling.

#%%[markdown]
#### Compute Cramér’s V matrix

#%%
cols = data_q3.columns
cv_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

for c1 in cols:
    for c2 in cols:
        cv_matrix.loc[c1, c2] = cramers_v(data_q3[c1], data_q3[c2])

# Visualization
vars_list = cv_matrix.columns
fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(cv_matrix.astype(float), cmap="coolwarm")

# Add numbers
for i in range(len(vars_list)):
    for j in range(len(vars_list)):
        ax.text(j, i, f"{cv_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)

ax.set_xticks(np.arange(len(vars_list)))
ax.set_yticks(np.arange(len(vars_list)))
ax.set_xticklabels(vars_list, rotation=90)
ax.set_yticklabels(vars_list)

plt.title("Cramér’s V Correlation Heatmap (Q3 Variables)")
plt.tight_layout()
plt.show()

#%%[markdown]
# None of the predictors show high correlation in the Cramér’s V matrix, and almost all values stay well below 0.60. 
# This indicates that there is no strong multicollinearity issue among the variables. 
# In addition, our analysis does not use linear regression models, so multicollinearity would not be a major concern in the first place. 
# Based on these results, all variables were kept for modeling.

#%%
# Prepare data for modeling

# Dependent variable
y = data_q3["Living Situation"]

# Independent variables
X = data_q3.drop(columns=["Living Situation"])

# One-hot encode X
X = pd.get_dummies(X, drop_first=True).astype(float)

# Encode y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

#%%[markdown]
#### Logistic Regression
#%%
# Setup Multinomial Logistic Regression
model_q3 = LogisticRegression(
    max_iter = 1000,
    solver = 'lbfgs',
    class_weight = 'balanced',
    multi_class = 'multinomial')

model_q3.fit(X_train, y_train)
y_pred = model_q3.predict(X_test)

# Results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = le.classes_))

#%%
# Visualization
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = le.classes_,
            yticklabels = le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Multinomial Logistic Regression")
plt.show()

#%%[markdown]
# The logistic regression model does not perform well, with an overall accuracy of about 0.49.
# The model predicts the majority class (Private Residence) with high precision (0.93)
# but still has low recall (0.47), and the smaller classes perform much worse,
# especially Institutional, which has extremely low precision (0.02).
# Because the data is highly imbalanced and the groups overlap a lot, the next step
# is to try more flexible models and apply SMOTE to improve the minority-class predictions.

#%%[markdown]
#### Random Forest

#%%
# Setup Random Forest

rf_model = RandomForestClassifier(
    n_estimators = 300,
    max_depth = None,
    class_weight = 'balanced_subsample',
    random_state = 42)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Results
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names = le.classes_))

#%%
# Visualization
cm_rf = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm_rf, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = le.classes_,
            yticklabels = le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Random Forest")
plt.show()

#%%[markdown]
# The random forest model performed better than logistic regression, with an overall accuracy of about 0.56. 
# For the largest group, Private Residence, precision stayed very high at 0.94, and recall was 0.55. 
# The Other Living Situation group also improved, with a recall of 0.56. 
# The Institutional Setting category remained the most difficult to predict. 
# Its precision was still extremely low (0.03), but recall increased to 0.64, mainly because the number of real cases in the test set is very small. 
# Overall, the random forest learned more structure than logistic regression, 
# but the severe class imbalance still limited how well the model could predict the minority group.

#%%[markdown]
#### SMOTE + Random Froest

#%%
# SMOTE + Random Forest


# Apply SMOTE to training data only
sm = SMOTE(random_state = 42, k_neighbors = 5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", pd.Series(y_train).value_counts())
print("After SMOTE:", pd.Series(y_train_sm).value_counts())

# Train Random Forest on SMOTE data
rf_sm = RandomForestClassifier(
    n_estimators = 300,
    max_depth = None,
    class_weight = "balanced_subsample",
    random_state = 42)

rf_sm.fit(X_train_sm, y_train_sm)
sm_pred = rf_sm.predict(X_test)

# Results
print(confusion_matrix(y_test, sm_pred))
print(classification_report(y_test, sm_pred, target_names = le.classes_))

#%%[markdown]
# After applying SMOTE to the training data, the random forest improved a bit. 
# The overall accuracy went up to about 0.58. For the largest group, Private Residence, 
# precision stayed high at 0.94 and recall increased to 0.58. The Other Living Situation group also performed better with a recall of 0.57. 
# The Institutional Setting category still had very low precision (0.03), but its recall increased to 0.60, 
# mainly because the test set contains very few actual cases from this group. 
# The weighted F1-score was about 0.65, so the SMOTE + random forest model ended up being the most balanced option among the ones we tried.

#%%[markdown]
#### Remove minority class with SMOTE + Random Froest
# After the presentation, we addressed the imbalance issue by removing the minority classes, 
# turning the Y variable into a binary outcome.

#%%
# Keep the class we need
data_q3_2class = data_q3[
    data_q3["Living Situation"].isin([
        "PRIVATE RESIDENCE",
        "OTHER LIVING SITUATION"])].copy()

#%%
# Dependent variable and indenpendent variable
y_last = data_q3_2class["Living Situation"]
X_last = data_q3_2class.drop(columns=["Living Situation"])

#%%
# One-hot encode X
X_last = pd.get_dummies(X_last, drop_first=True).astype(float)

# Encode y
le2 = LabelEncoder()
y_last_encoded = le2.fit_transform(y_last)

#%%
# Train-test split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_last, y_last_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_last_encoded)

#%%
# Apply SMOTE
sm2 = SMOTE(random_state=42, k_neighbors=5)
X2_train_sm, y2_train_sm = sm2.fit_resample(X2_train, y2_train)

#%%
rf2_sm = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced_subsample",
    random_state=42
)

rf2_sm.fit(X2_train_sm, y2_train_sm)
y2_pred_sm = rf2_sm.predict(X2_test)

print(confusion_matrix(y2_test, y2_pred_sm))
print(classification_report(y2_test, y2_pred_sm, target_names=le2.classes_))

#%%[markdown]
# The model reached an accuracy of 0.67.
# For Private Residence, precision was high at 0.93 and recall was 0.64, so the model predicts this group fairly well.
# For Other Living Situation, recall was 0.80, but precision was lower at 0.33, meaning many predictions were incorrect.
# The weighted F1-score was 0.70, which shows the overall performance is acceptable after switching to a binary target.
# Compared to the earlier multi-class results, the binary version performs better, especially in recall.

#%%[markdown]
#### Feature Importance
# %%

importances2 = rf2_sm.feature_importances_
feature_names2 = X_last.columns

imp2_df = pd.DataFrame({
    "Feature": feature_names2,
    "Importance": importances2
}).sort_values(by="Importance", ascending=False)

# Plot Top 15 Features
plt.barh(imp2_df.head(15)["Feature"], imp2_df.head(15)["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance Level")
plt.title("Top 15 Important Features (Random Forest + Binary Target)")
plt.show()

#%%[markdown]
# Our results show that financial support and insurance coverage are closely related to a person’s living situation. 
# Age is the strongest factor in the model. People in different age groups tend to face different levels of financial pressure, access to benefits, and stability in housing, which helps explain why age stands out so clearly in the feature importance ranking. 
# Insurance variables such as private insurance, Medicaid, and Medicare also appear near the top of the importance ranking. 
# Cash assistance programs including SSI and public assistance further contribute to predicting living stability.
# These findings match what previous studies have reported. Fenelon et al. (2017) found that housing assistance helps people maintain stable living conditions. 
# Friedman et al. (2022) observed that people who rely on Medicaid often experience more housing instability. 
# The patterns in our model are consistent with these observations because insurance and cash benefits play major roles in determining living situations.

#%%[markdown]
#### Conclusion for Q3

# In Q3, predicting living situation was challenging because the classes were highly imbalanced and the patterns between groups were very similar. 
# Logistic regression did not perform well with an accuracy of about 0.49. The random forest model did slightly better, but it still struggled to identify the smallest class. 
# After applying SMOTE, the random forest improved to an accuracy of around 0.56 and became the most balanced model among the ones we tested.
# We also tested an equal-size sampling method by reducing the larger groups to the size of the Institutional group, but this caused a major loss of data and made the model unstable.
# Based on these results, the SMOTE random forest was the most practical choice for Q3. 
# The feature importance results showed that age was the strongest predictor, followed by insurance coverage such as private insurance, Medicaid, and Medicare. 
# Cash assistance programs, including SSI and public assistance, also played an important role. 
# Overall, these factors helped the model distinguish between different living situations more effectively than other variables.



#%% [markdown]
## 5. Conclusion
# Across the three smart questions, the PCS dataset helped reveal clear patterns linking socioeconomic and demographic factors with both mental-illness reporting and living-situation categories. Although data imbalance limited prediction strength, the analyses consistently identified which variables had the greatest influence on each outcome.
#
# For mental illness (Q1 & Q2), age group, education, household composition, employment status, and several insurance/assistance variables showed meaningful associations, indicating that social and economic stability is linked with how individuals report mental-health conditions.
# For living situation (Q3), age group, insurance coverage, cash-assistance programs, and employment status were the strongest predictors, helping separate private residence, institutional settings, and other living arrangements.
#
# Overall, the models highlighted the importance of insurance, financial support, age, and employment as the most consistent drivers across all three questions, providing useful insight into how socioeconomic conditions shape both mental-health outcomes and housing patterns in New York State.


#%% [markdown]
## 6. Recommendations
# A main limitation in this project is the severe class imbalance across all three questions, 
# which makes the smaller groups difficult for the models to learn and leads to unstable results. 
# Another issue is that most variables in the PCS dataset are categorical and split into many small levels, 
# increasing the chance of overfitting and limiting the models’ ability to capture stronger patterns.
#
# For the next step, we could try more advanced imbalance-handling methods such as Balanced Random Forest or XGBoost with class weights to improve the performance for minority groups. 
# Adding continuous variables, if they become available, may help strengthen the predictive signal. 
# It may also be useful to analyze minority groups separately so their patterns are not overshadowed by the dominant majority class.


#%% [markdown]
## 7. References
# Fenelon, A., Slopen, N., Boudreaux, M., & Pollack, C. E. (2017). *Housing assistance programs and adult health in the United States.* American Journal of Public Health, 107(4), 571–578. https://doi.org/10.2105/AJPH.2016.303649
#
# Friedman, C., Schiro, S., & Remmert, J. E. (2022). *Housing insecurity of Medicaid beneficiaries with cognitive disabilities during the COVID-19 pandemic.* Journal of Social Distress and the Homeless. Advance online publication. https://doi.org/10.1080/10530789.2022.2096710
#
# Healthy People 2030. (n.d.). *Housing instability.* Office of Disease Prevention and Health Promotion. https://health.gov/
#
# Maqbool, N., Viveiros, J., & Ault, M. (2015). *The impacts of affordable housing on health: A research summary.* National Housing Conference.
#
# Tanarsuwongkul, S., et al. (2025). Associations between social determinants of health and mental health disorders among US adults. *Epidemiology and Psychiatric Sciences.*