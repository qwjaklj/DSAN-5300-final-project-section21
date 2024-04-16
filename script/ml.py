import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import seaborn as sns

# Load data
file_path = '../data/data_cleaned.csv'
data = pd.read_csv(file_path)

# Fill missing values for numeric columns
data = data.fillna(data.mean(numeric_only=True))
#data = data.dropna()

# Selected features from feature selection
selected_features = ['SPIN14','SPIN15','SPIN13','SPIN6', 'Age', 'Hours','SWL1', 'SWL3', 'SWL4']
X = data[selected_features]

# Target for regression and classification
y = data['GAD_T']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert regression target to classification target
y_train_clf = pd.qcut(y_train, q=4, labels=False)
y_test_clf = pd.qcut(y_test, q=4, labels=False)

# Binarize the output classes for ROC curve
y_train_clf_binarized = label_binarize(y_train_clf, classes=[0, 1, 2, 3])
y_test_clf_binarized = label_binarize(y_test_clf, classes=[0, 1, 2, 3])
n_classes = y_train_clf_binarized.shape[1]

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=100),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Classifier': SVC(probability=True, random_state=42)
}

classification_metrics = {}
regression_metrics = {}
# Train classifiers and generate ROC curves
for clf_name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train_clf)
    y_pred_clf = clf.predict(X_test_scaled)
    y_score = clf.predict_proba(X_test_scaled)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_clf_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_clf_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{clf_name} Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig(f'../ml_image/{clf_name}_ROC.png')
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues')  # Using percentage format
    plt.title(f'{clf_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'../ml_image/{clf_name}_CM.png')
    plt.close()

# Linear Regression does not need ROC as it is not a classification model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

# Plot residuals for linear regression
plt.figure()
plt.scatter(y_test, y_test - y_pred_reg, color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.title('Residual errors')
plt.savefig(f'../ml_image/LinearRegression_Residuals.png')
plt.close()