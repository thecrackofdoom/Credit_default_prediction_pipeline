# Databricks notebook source
# DBTITLE 1,Install ML dependencies
# MAGIC %pip install -q plotly imblearn xgboost scikit-learn scipy seaborn matplotlib mlflow shap --quiet

# COMMAND ----------

# DBTITLE 1,Import libraries
# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, roc_auc_score, roc_curve, f1_score
)
import xgboost as xgb
from xgboost import XGBClassifier

# MLflow and SHAP
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap

# Set plotting theme
pio.templates.default = "plotly_dark"
sns.set_style('darkgrid')

# COMMAND ----------

# DBTITLE 1,Load gold table from pipeline
# Load the feature-engineered gold table from the pipeline
# This table already has:
# - ID dropped
# - Categorical features (SEX, EDUCATION, MARRIAGE) one-hot encoded
# - All numeric features and target (de_pay) ready for ML

df_gold = spark.table('workspace.end_to_end_credit_default.uci_credit_card_gold')
df = df_gold.toPandas()

print(f"Gold table shape: {df.shape}")
print(f"Target distribution:\n{df['de_pay'].value_counts()}")
print(f"\nDefault rate: {df['de_pay'].mean():.2%}")
display(df.head())

# COMMAND ----------

# DBTITLE 1,Prepare features and target
# Split features and target
X = df.drop('de_pay', axis=1)
y = df['de_pay']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns:\n{X.columns.tolist()}")

# COMMAND ----------

# DBTITLE 1,Train/test split (stratified)
# Stratified train/test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y, 
    random_state=100
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTrain target distribution:\n{y_train.value_counts()}")
print(f"\nTest target distribution:\n{y_test.value_counts()}")

# COMMAND ----------

# DBTITLE 1,Baseline: Logistic Regression with StandardScaler
# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
param_lr = {'max_iter': 1000, 'solver': 'lbfgs', 'C': 1}
LR = LogisticRegression(**param_lr, random_state=100)
LR.fit(X_train_scaled, y_train)

# Predictions
y_pred = LR.predict(X_test_scaled)
LR_report = classification_report(y_test, y_pred, output_dict=True)

# Evaluation
print('=== Logistic Regression Results ===')
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'\n{classification_report(y_test, y_pred)}')

# Cross-validation
cv_scores = cross_val_score(LR, X_train_scaled, y_train, cv=5)
print(f"\nAverage 5-Fold CV Score: {np.mean(cv_scores):.4f}")
print(f"Standard deviation: {np.std(cv_scores):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
CMdist = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
CMdist.plot()
plt.title('Logistic Regression - Confusion Matrix')
plt.show()

# COMMAND ----------

# DBTITLE 1,Prepare oversampling strategies
# Strategy 1: Normal sampling (baseline - imbalanced)
X_train_normal = X_train
y_train_normal = y_train

# Strategy 2: Manual oversampling (resample minority class)
df_train = X_train.join(y_train)
df_majority = df_train[df_train.de_pay == 0]
df_minority = df_train[df_train.de_pay == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
X_train_oversample = df_upsampled.drop('de_pay', axis=1)
y_train_oversample = df_upsampled['de_pay']

print(f"Oversampled training set: {y_train_oversample.value_counts()}")

# Strategy 3: SMOTE (synthetic minority oversampling)
sm = SMOTE(random_state=100)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print(f"\nSMOTE training set: {pd.Series(y_train_smote).value_counts()}")

# Package all strategies
xtrain_data = [X_train_normal, X_train_oversample, X_train_smote]
ytrain_data = [y_train_normal, y_train_oversample, y_train_smote]
strategy_names = ['Normal Sampling', 'Manual Oversampling', 'SMOTE']

# COMMAND ----------

# DBTITLE 1,XGBoost model evaluation function
# Store results for MLflow logging
XGB_reports = []
XGB_models = []

def model_eval(algo, Xtrain, ytrain, Xtest, ytest, strategy_name):
    """Train and evaluate XGBoost model with comprehensive metrics"""
    
    # Train model
    algo.fit(Xtrain, ytrain)
    
    # Training metrics
    y_train_pred = algo.predict(Xtrain)
    y_train_prob = algo.predict_proba(Xtrain)[:, 1]
    train_accuracy = accuracy_score(ytrain, y_train_pred)
    train_auc = roc_auc_score(ytrain, y_train_prob)
    
    # Test metrics
    y_test_pred = algo.predict(Xtest)
    y_test_prob = algo.predict_proba(Xtest)[:, 1]
    test_accuracy = accuracy_score(ytest, y_test_pred)
    test_auc = roc_auc_score(ytest, y_test_prob)
    
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_accuracy:.4f} | Train AUC: {train_auc:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f} | Test AUC:  {test_auc:.4f}")
    print(f"\nClassification Report (Test):")
    print(classification_report(ytest, y_test_pred))
    
    # K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(Xtrain, ytrain):
        X_kfold_train, X_kfold_val = Xtrain.iloc[train_idx], Xtrain.iloc[val_idx]
        y_kfold_train, y_kfold_val = ytrain.iloc[train_idx], ytrain.iloc[val_idx]
        algo.fit(X_kfold_train, y_kfold_train)
        y_kfold_pred = algo.predict(X_kfold_val)
        scores.append(roc_auc_score(y_kfold_val, y_kfold_pred))
    
    print(f"\n5-Fold CV AUC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion matrix
    cm = confusion_matrix(ytest, y_test_pred)
    sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt="d",
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                linewidths=0.5, ax=ax[0])
    ax[0].set_ylabel('True Label')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_title(f'Confusion Matrix - {strategy_name}')
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(ytest, y_test_prob)
    ax[1].plot(fpr, tpr, color='r', label=f'AUC = {test_auc:.4f}')
    ax[1].plot([0, 1], [0, 1], color='green', linestyle='--', label='Random')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_title(f'ROC Curve - {strategy_name}')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Store for MLflow logging
    report = classification_report(ytest, y_test_pred, output_dict=True)
    XGB_reports.append(report)
    XGB_models.append(algo)
    
    return report

# COMMAND ----------

# DBTITLE 1,XGBoost hyperparameters (tuned)
# Pre-tuned hyperparameters for each sampling strategy
# These were optimized through RandomizedSearchCV
param_xgb = [
    # Normal sampling - higher regularization due to imbalance
    {
        'min_child_weight': 5,
        'max_depth': 4,
        'learning_rate': 0.1,
        'gamma': 0.2,
        'colsample_bytree': 0.4,
        'scale_pos_weight': 3
    },
    # Manual oversampling - deeper trees allowed
    {
        'min_child_weight': 1,
        'max_depth': 15,
        'learning_rate': 0.2,
        'gamma': 0.1,
        'colsample_bytree': 0.4
    },
    # SMOTE - moderate depth with regularization
    {
        'min_child_weight': 1,
        'max_depth': 15,
        'learning_rate': 0.05,
        'gamma': 0.4,
        'colsample_bytree': 0.4
    }
]

print("Hyperparameters ready for 3 strategies:")
for name, params in zip(strategy_names, param_xgb):
    print(f"\n{name}:")
    for k, v in params.items():
        print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Train XGBoost models with all strategies
# Train and evaluate XGBoost with each sampling strategy
for X_train_data, y_train_data, strategy_name, params in zip(
    xtrain_data, ytrain_data, strategy_names, param_xgb
):
    print(f"\n\n{'#'*80}")
    print(f"Training XGBoost with: {strategy_name}")
    print(f"Hyperparameters: {params}")
    print(f"{'#'*80}")
    
    xgb_model = XGBClassifier(**params, random_state=100)
    model_eval(xgb_model, X_train_data, y_train_data, X_test, y_test, strategy_name)

# COMMAND ----------

# DBTITLE 1,MLflow: Log Logistic Regression
# Set MLflow experiment (uses workspace MLflow by default)
mlflow.set_experiment("/Users/tranminhnguyen1106@gmail.com/Credit Default Prediction")

# Log Logistic Regression baseline
with mlflow.start_run(run_name="Logistic Regression (Baseline)"):
    mlflow.log_params(param_lr)
    mlflow.log_params({'scaler': 'StandardScaler'})
    
    mlflow.log_metrics({
        'accuracy': LR_report['accuracy'],
        'precision_macro': LR_report['macro avg']['precision'],
        'recall_macro': LR_report['macro avg']['recall'],
        'f1_macro': LR_report['macro avg']['f1-score'],
        'recall_no_default': LR_report['0']['recall'],
        'recall_default': LR_report['1']['recall']
    })
    
    mlflow.sklearn.log_model(LR, "logistic_regression_model")
    print("✓ Logistic Regression logged to MLflow")

# COMMAND ----------

# DBTITLE 1,MLflow: Log all XGBoost models
# Log all XGBoost models to MLflow
for report, model, strategy_name, params in zip(
    XGB_reports, XGB_models, strategy_names, param_xgb
):
    model_name = f"XGBoost - {strategy_name}"
    
    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param('sampling_strategy', strategy_name)
        
        # Log metrics
        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'recall_no_default': report['0']['recall'],
            'recall_default': report['1']['recall']
        })
        
        # Log model
        mlflow.xgboost.log_model(model, f"xgboost_{strategy_name.lower().replace(' ', '_')}")
        print(f"✓ {model_name} logged to MLflow")

print("\n✅ All models logged to MLflow successfully!")

# COMMAND ----------

# DBTITLE 1,SHAP: Explain individual predictions
# SHAP explanations for the first XGBoost model (Normal Sampling)
best_model = XGB_models[0]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Explain a specific prediction
sample_index = 2
sample_prediction = best_model.predict(X_test.iloc[[sample_index]])[0]
actual_label = y_test.iloc[sample_index]

print(f"Sample #{sample_index}:")
print(f"  Actual Label: {actual_label} ({'Default' if actual_label == 1 else 'No Default'})")
print(f"  Predicted: {sample_prediction} ({'Default' if sample_prediction == 1 else 'No Default'})")
print(f"\n{'='*80}\n")

# Force plot (individual prediction breakdown)
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_test.iloc[sample_index],
    feature_names=X_test.columns.tolist(),
    show=False,
    matplotlib=True
)
plt.tight_layout()
plt.show()

print(f"\n{'='*80}\n")

# Waterfall plot (cascading feature contributions)
plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_index],
        feature_names=X_test.columns.tolist()
    ),
    max_display=10,
    show=False
)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,SHAP: Global feature importance
# SHAP summary plot - shows feature importance across all predictions
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X_test.columns.tolist(),
    max_display=15,
    show=False,
    plot_type='bar'
)
plt.title('SHAP Feature Importance (Global)')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,XGBoost native feature importance
# XGBoost's built-in feature importance
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance_df.head(15).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance (Top 15)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Model performance metrics: KS & Gini
def KS_statistic(y_true, y_pred_prob):
    """Kolmogorov-Smirnov statistic from ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    ks_statistic = max(tpr - fpr)
    return ks_statistic

def KS_stat_scipy(y_true, y_pred_prob):
    """KS statistic using scipy's two-sample test"""
    import scipy.stats as stats
    prob_positive = y_pred_prob[y_true == 1]
    prob_negative = y_pred_prob[y_true == 0]
    return stats.ks_2samp(prob_positive, prob_negative).statistic

def gini_coefficient(y_true, y_pred_prob):
    """Gini coefficient from AUC"""
    auc = roc_auc_score(y_true, y_pred_prob)
    return 2 * auc - 1

# Calculate for best XGBoost model
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

ks_val_1 = KS_statistic(y_test, y_pred_prob)
ks_val_2 = KS_stat_scipy(y_test, y_pred_prob)
gini_val = gini_coefficient(y_test, y_pred_prob)

print("=" * 60)
print("Model Performance Metrics (XGBoost - Normal Sampling)")
print("=" * 60)
print(f"KS Statistic (ROC-based):  {ks_val_1:.4f}")
print(f"KS Statistic (scipy):      {ks_val_2:.4f}")
print(f"Gini Coefficient:          {gini_val:.4f}")
print("=" * 60)

# Interpretation guide
print("\nInterpretation:")
print("  KS Statistic: Higher is better (0-1 range). >0.4 is good for credit scoring.")
print("  Gini Coefficient: Higher is better (0-1 range). >0.6 is good for credit models.")