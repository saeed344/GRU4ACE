import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import math
from imblearn.over_sampling import SMOTE
import joblib


# Cross-validation function
def cv(clf, X, y, nr_fold):
    ix = np.arange(len(y))
    allACC, allSENS, allSPEC, allMCC, allAUC, allF1 = [], [], [], [], [], []
    probtest, testsample = np.array([]), np.array([])

    for j in range(nr_fold):
        train_ix = (ix % nr_fold) != j
        test_ix = (ix % nr_fold) == j
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix].ravel()

        clf.fit(train_X, train_y)
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:, 1]

        probtest = np.concatenate((probtest, pr))
        testsample = np.concatenate((testsample, test_y))

        TP = np.sum((test_y == 1) & (p == 1))
        FP = np.sum((test_y == 0) & (p == 1))
        TN = np.sum((test_y == 0) & (p == 0))
        FN = np.sum((test_y == 1) & (p == 0))

        ACC = (TP + TN) / (TP + FP + TN + FN)
        SENS = TP / (TP + FN)
        SPEC = TN / (TN + FP)
        MCC = 0 if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0 else ((TP * TN) - (
                FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        AUC = roc_auc_score(test_y, pr)
        F1 = f1_score(test_y, p)

        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
        allF1.append(F1)

    return np.mean(allACC), np.mean(allSENS), np.mean(allSPEC), np.mean(allMCC), np.mean(allAUC), np.mean(
        allF1), probtest, testsample


# Test function
def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt.ravel()

    clf.fit(train_X, train_y)
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:, 1]

    TP = np.sum((test_y == 1) & (p == 1))
    FP = np.sum((test_y == 0) & (p == 1))
    TN = np.sum((test_y == 0) & (p == 0))
    FN = np.sum((test_y == 1) & (p == 0))

    ACC = (TP + TN) / (TP + FP + TN + FN)
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)
    MCC = 0 if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0 else ((TP * TN) - (FP * FN)) / math.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    AUC = roc_auc_score(test_y, pr)
    F1 = f1_score(test_y, p)

    return ACC, SENS, SPEC, MCC, AUC, F1, pr, test_y


# Load and preprocess data
data_ = pd.read_csv('EN_All_clean.csv', header=None)
data_np = np.array(data_)
data = data_np[:, :]

label1 = np.ones((394, 1))
label2 = np.zeros((626, 1))
labels = np.append(label1, label2)

# Balance dataset with SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(data, labels)

# Split data into training and independent sets
X_train, X_ind, y_train, y_ind = train_test_split(X, y, test_size=0.2, random_state=42)

X = np.array(X_train, dtype=float)
y = y_train
Xt = X_ind
yt = y_ind

allclf = []

# Prepare to write classifier results
file = open("Classifier_cv_results.csv", "w")

# Classifier parameters and evaluation
param_grids = {
    "SVM": {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
    "XGB": {'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]},
    "RF": {'n_estimators': [50, 100, 200, 500]},
    "ET": {'n_estimators': [50, 100, 200, 500]},
    "MLP": {'hidden_layer_sizes': [(50,), (100,), (200,), (500,)]},
    "NB": {},
    "DT": {},
    "1NN": {'n_neighbors': [1]},
    "LR": {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
}

# Grid Search and training
for clf_name, param_grid in param_grids.items():
    if clf_name == "SVM":
        model = SVC(probability=True)
    elif clf_name == "XGB":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif clf_name == "RF":
        model = RandomForestClassifier()
    elif clf_name == "ET":
        model = ExtraTreesClassifier()
    elif clf_name == "MLP":
        model = MLPClassifier()
    elif clf_name == "NB":
        model = GaussianNB()
    elif clf_name == "DT":
        model = DecisionTreeClassifier()
    elif clf_name == "1NN":
        model = KNeighborsClassifier()
    elif clf_name == "LR":
        model = LogisticRegression()

    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=1, verbose=1)
    grid_search.fit(X, y)

    best_clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    acc, sens, spec, mcc, roc, F1, probtest, testsample = cv(best_clf, X, y, 5)

    allclf.append(best_clf)
    param_str = ', '.join([f"{key}={value}" for key, value in best_params.items()]) if best_params else "N/A"
    file.write(f"{clf_name},{acc},{sens},{spec},{mcc},{roc},{F1},{param_str}\n")

    # Save probabilities and test labels
    data_csv = pd.DataFrame(data=probtest)
    data_csv.to_csv(f'train_{clf_name}_prob.csv', index=False)
    data_csv = pd.DataFrame(data=testsample)
    data_csv.to_csv(f'train_{clf_name}_labelTest.csv', index=False)

file.close()

# Test classifiers on independent data
file = open("classifier_test_results.csv", "w")
for i, clf in enumerate(allclf):
    clf_name = list(param_grids.keys())[i]
    acc, sens, spec, mcc, roc, F1, prob, label = test(clf, X, y, Xt, yt)
    file.write(f"{clf_name},{acc},{sens},{spec},{mcc},{roc},{F1}\n")

    data_csv = pd.DataFrame(data=prob)
    data_csv.to_csv(f'test_{clf_name}_prob.csv', index=False)
    data_csv = pd.DataFrame(data=label)
    data_csv.to_csv(f'test_{clf_name}_labelTest.csv', index=False)

file.close()
