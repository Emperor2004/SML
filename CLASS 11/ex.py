# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_auc_score, roc_curve

# %%
# Load dataset
df = pd.read_csv('Practice dataset BankChurners.csv')
df.head()

# %%
# Dataset overview
df.shape
# %%
df.info()
# %%
df.isna().sum()
# %%
df.describe(include='all')
# %%
df.dtypes

# %%
# Encode categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
print(df.head())

# %%
# Define features and target variables
columns_to_drop = [
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
]
X = df.drop(columns=columns_to_drop)
y1 = df[columns_to_drop[0]]
y2 = df[columns_to_drop[1]]

# %%
# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42, stratify=y1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)

# %%
# Standardize features
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)
X2_train = scaler.transform(X2_train)
X2_test = scaler.transform(X2_test)

# %%
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X1_train, y1_train = smote.fit_resample(X1_train, y1_train)
X2_train, y2_train = smote.fit_resample(X2_train, y2_train)

# %%
# Train models
classifier1 = GaussianNB()
classifier1.fit(X1_train, y1_train)

classifier2 = GaussianNB()
classifier2.fit(X2_train, y2_train)

# %%
# Predictions
y1_pred = classifier1.predict(X1_test)
y2_pred = classifier2.predict(X2_test)

# %%
# Evaluation
def evaluate_model(y_test, y_pred, title):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{title} - GaussianNB Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

evaluate_model(y1_test, y1_pred, 'Model 1')
evaluate_model(y2_test, y2_pred, 'Model 2')

# %%
# Confusion Matrices
def plot_confusion_matrix(y_test, y_pred, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

plot_confusion_matrix(y1_test, y1_pred, "Confusion Matrix - Model 1")
plot_confusion_matrix(y2_test, y2_pred, "Confusion Matrix - Model 2")

# %%
# ROC-AUC Curves
def plot_roc_curve(y_test, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

y1_pred_proba = classifier1.predict_proba(X1_test)[:, 1]
y2_pred_proba = classifier2.predict_proba(X2_test)[:, 1]

plot_roc_curve(y1_test, y1_pred_proba, "ROC-AUC Curve - Model 1")
plot_roc_curve(y2_test, y2_pred_proba, "ROC-AUC Curve - Model 2")

# %%
# Final ROC-AUC Scores
roc_auc_y1 = roc_auc_score(y1_test, y1_pred)
print("ROC AUC Score - Model 1:", roc_auc_y1)

roc_auc_y2 = roc_auc_score(y2_test, y2_pred)
print("ROC AUC Score - Model 2:", roc_auc_y2)
