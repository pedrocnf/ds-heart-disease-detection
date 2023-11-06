import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# Normality test to detect if any column of a dataframe have normal-distribution
def normality_test(df, colunas):
    results = []
    for col in colunas:  # Exclua a coluna 'class'
        stat, p_value = stats.shapiro(df[col])
        is_normal = p_value > 0.05  # Verifica se os dados são considerados normais (p-valor > 0.05)
        results.append([col, p_value, is_normal])
    
    return pd.DataFrame(results, columns=["Coluna", "p_valor", "Normal?"])

# Mann-whitney test for differences:
def differences_test(grupo1, grupo2, colunas):
    results = []
    for col in colunas:  # Exclua a coluna 'class'
        stat, p_value = stats.ttest_ind(grupo1[col], grupo2[col])
        is_different = p_value > 0.05  # Verifica se os dados são considerados normais (p-valor > 0.05)
        results.append([col, p_value, is_different])
    
    return pd.DataFrame(results, columns=["Coluna", "p_valor", "Different?"])

# Model execution
def model_evaluation(params, X_train, y_train, X_test, y_test):
    metrics_df = pd.DataFrame(columns=['Accuracy', 'F1-Score', 'Recall', 'Precision', 'AUC-ROC'])
    if params == None:
        model = lgb.LGBMClassifier(random_state = 42)
    else:
        model = lgb.LGBMClassifier(**params, random_state = 42)
    model.fit(X_train, y_train, eval_set=[(X_test,y_test),(X_train,y_train)])
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] 
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    metrics_df = metrics_df.append({
    'Accuracy': accuracy,
    'F1-Score': f1,
    'Recall': recall,
    'Precision': precision,
    'AUC-ROC': auc_roc}, ignore_index=True)

    lgb.plot_metric(model)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
  
    RocCurveDisplay.from_predictions(
    y_pred,
     y_test,
    name=f"Not fraud vs Fraud",
    color="darkorange",
    plot_chance_level=True)
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fraud vs Not Fraud)")
    plt.legend()
    
    plt.show()

    return metrics_df
    