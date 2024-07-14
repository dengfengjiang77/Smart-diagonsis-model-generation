#%pip install optuna reportlab
import json
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import optuna
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, when, count, split
from functools import reduce

spark = SparkSession.builder.appName("Filter").getOrCreate()


## Get config file（input str output dict）
def getConfig(configFilePath):
    with open(configFilePath) as f:
        config = json.load(f)
    return config

## Defining Data Conversion Functions - helper function for button2
def convert_to_number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

### Data Processing ###

## Filter Claim Data with tnf
def getclaimnum(select_com,component,claim_df_path,tnf_path):
    claim_df = spark.read.table(claim_df_path)
    tnf = spark.read.table(tnf_path).select('Claim_number','TNF_Flag').dropna()
    filtered_df = claim_df.where(claim_df.ABO_FAILURE_PART.isin(component)).cache()
    filtered_df = filtered_df.withColumn('label', when(claim_df.ABO_FAILURE_PART.isin(select_com), 1).otherwise(0)).cache()
    filtered_df = filtered_df.select('Claim_number','ESN','ABO_FAILURE_PART','Failure_date','label')
    filtered_df = filtered_df.join(tnf,on='Claim_number',how='left')
    filtered_df = filtered_df[filtered_df['TNF_Flag']!='Y']
    filtered_df = filtered_df.drop('TNF_Flag')
    # Drop duplicate
    counts = filtered_df.groupBy("ESN").agg(count("ESN").alias("count"))
    filtered_df = filtered_df.join(counts, "ESN")
    filtered_df = filtered_df.filter(col("count") == 1).drop("count")
    return filtered_df
## Get eFPA feature
def getefpa(unchanged_feature_list,feature_list,efpa_path,eng_condition1,eng_condition2):
    all_feature = unchanged_feature_list + feature_list
    efpa = spark.read.table(efpa_path).select(all_feature)
    efpa_filter = efpa.filter((split(col('engine_model_group'), '_')[0].isin(eng_condition1)) &  # In China eFPA Engine Family format is "platform_emissionlevel_manufacturer"
        (split(col('engine_model_group'), '_')[1].isin(eng_condition2))).cache()
    return efpa_filter
## filter eFPA with FC in 7d
def filterefpa(claim_df, efpa_df, fc_list):
    claim_with_efpa = claim_df.join(efpa_df, "ESN")

    claim_with_efpa_7d = claim_with_efpa.filter(
        (col("Failure_date") >= col("Occurrence_time")) & 
        (datediff(col("Failure_date"), col("Occurrence_time")) <= 7)
    ).cache()

    conditions = [col("Faultlist").contains(f"[{fc}]") | col("Faultlist2").contains(f"[{fc}]") for fc in fc_list]
    condition = reduce(lambda x, y: x | y, conditions)
    filtered_df = claim_with_efpa_7d.filter(condition).cache()

    grouped_df = filtered_df.groupBy("ESN").count()
    final_df = claim_with_efpa_7d.join(grouped_df.select("ESN"), "ESN").cache()
    return final_df


### pre-parameter importance ranking ###
def ranking_xgboost(X, y, split_ratio, random_seed, early_stopping_rounds, max_depth, max_iterations, save_path):
    # split data into train and test sets Divide the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)

    # training model
    model = xgb.XGBClassifier(
        max_depth=max_depth, 
        n_estimators=max_iterations, 
        random_state=random_seed,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="auc"
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    
    # Getting feature importance
    importance = model.get_booster().get_score(importance_type='weight')
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    importance_sorted = [item[0] for item in importance_sorted]
    top_10_features = importance_sorted[:10]


    # Create and save feature importance maps
    plt.figure(figsize=(10, 5))
    xgb.plot_importance(model)
    plt.title('Feature Importance')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path, 'XGBoost_feature_importance.png'), dpi=300)
    plt.show()
    plt.close()

    # Importance of Printing Top 10 Characteristics
    print("Top 10 feature importances:") 
    for feature in top_10_features:
        print(feature)    
        
    return model, top_10_features, importance_sorted



### Model training and evaluation ###
def train_evaluate_xgboost(X, y, split_ratio, random_seed, early_stopping_rounds, max_depth, max_iterations, save_path):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)

    # training model
    model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=max_iterations, random_state=random_seed)
    model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric="auc", eval_set=[(X_test, y_test)], verbose=False)

    # predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    thresholds_every = round(len(tpr)/10)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(0, len(thresholds), thresholds_every):
        plt.text(fpr[i], tpr[i], np.round(thresholds[i], 2), fontsize=9)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cure for XGBOOST Model')
    plt.legend(loc="lower right")
    # plt.show()

    # Save the plot
    roc_save_path = os.path.join(save_path, 'XGBoost_roc_curve.png')
    plt.savefig(roc_save_path, dpi=300)
    plt.show()
    plt.close()

    # Fit SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Visualization SHAP
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    # Save the plot
    shap_save_path = os.path.join(save_path, 'XGBoost_shap_summary.png')
    plt.savefig(shap_save_path, dpi=300)
    plt.show()
    plt.close()
    # generate a dataframe to store the thresholds, fpr and tpr, only fpr less than 0.2
    df_thresholds = pd.DataFrame({
    'Threshold': thresholds,
    'FPR': fpr,
    'TPR': tpr
})  
    df_thresholds = df_thresholds[df_thresholds['FPR'] <= 0.2]
    print(df_thresholds)
    return model, df_thresholds


def train_random_forest(X, y, split_ratio, random_seed, max_depth, max_iterations, save_path):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)

    # training model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=max_iterations, random_state=random_seed)
    model.fit(X_train, y_train)

    # predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    thresholds_every = round(len(tpr)/10)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(0, len(thresholds), thresholds_every):
        plt.text(fpr[i], tpr[i], np.round(thresholds[i], 2), fontsize=9)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cure for Random Forest Model')
    plt.legend(loc="lower right")

    # Save the plot
    roc_save_path = os.path.join(save_path, 'Random_Forest_roc_curve.png')
    plt.savefig(roc_save_path, dpi=300)
    plt.show()
    plt.close()

    # Visualization SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values[1], X_train, show=False)
    # Save the plot
    shap_save_path = os.path.join(save_path, 'Random_Forest_shap_summary.png')
    plt.savefig(shap_save_path, dpi=300)
    plt.show()
    plt.close()
    # generate a dataframe to store the thresholds, fpr and tpr, only fpr less than 0.2
    df_thresholds = pd.DataFrame({
    'Threshold': thresholds,
    'FPR': fpr,
    'TPR': tpr
})  
    df_thresholds = df_thresholds[df_thresholds['FPR'] <= 0.2]
    print(df_thresholds)

    return model, df_thresholds

## optuna
def objective_xgb(trial, X, y, split_ratio, random_seed, early_stopping_rounds):
    # Suggest parameters
    max_depth = trial.suggest_int('max_depth', 2, 10)
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)

    # Training model
    model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_seed, use_label_encoder=False)
    model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric="auc", eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate model
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    return roc_auc

def train_evaluate_xgboost_with_optuna(X, y, save_path, split_ratio=0.2, random_seed=42, early_stopping_rounds=10, n_trials=10):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_xgb(trial, X, y, split_ratio, random_seed, early_stopping_rounds), n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    
    # Retrain model with best parameters
    model = xgb.XGBClassifier(**best_params, random_state=random_seed, use_label_encoder=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)
    model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric="auc", eval_set=[(X_test, y_test)], verbose=True)
    
    # predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    thresholds_every = round(len(tpr)/10)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(0, len(thresholds), thresholds_every):
        plt.text(fpr[i], tpr[i], np.round(thresholds[i], 2), fontsize=9)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cure for XGBOOST Model')
    plt.legend(loc="lower right")

    # Save the plot
    roc_save_path = os.path.join(save_path, 'XGBoost_roc_curve.png')
    plt.savefig(roc_save_path, dpi=300)
    plt.show()
    plt.close()

    # Fit SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Visualization SHAP
    shap.summary_plot(shap_values, X_train, show=False)
    # Save the plot
    shap_save_path = os.path.join(save_path, 'XGBoost_shap_summary.png')
    plt.savefig(shap_save_path, dpi=300)
    plt.show()
    plt.close()
    # generate a dataframe to store the thresholds, fpr and tpr, only fpr less than 0.2
    df_thresholds = pd.DataFrame({
    'Threshold': thresholds,
    'FPR': fpr,
    'TPR': tpr
})  
    df_thresholds = df_thresholds[df_thresholds['FPR'] <= 0.2]
    print(df_thresholds)
    return model, study, df_thresholds

def objective_rf(trial, X, y, split_ratio, random_seed):
    # Suggest parameters
    max_depth = trial.suggest_int('max_depth', 2, 32)
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)
    
    # Training model
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_seed)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    
    return roc_auc

def train_random_forest_with_optuna(X, y, save_path, split_ratio=0.2, random_seed=42, n_trials=10):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_rf(trial, X, y, split_ratio, random_seed), n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    
    # Retrain model with best parameters
    model = RandomForestClassifier(**best_params, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_seed)
    model.fit(X_train, y_train)
    
    # Visualization and analysis as before
    # predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    thresholds_every = round(len(tpr)/10)

    # plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(0, len(thresholds), thresholds_every):
        plt.text(fpr[i], tpr[i], np.round(thresholds[i], 2), fontsize=9)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cure for Random Forest Model')
    plt.legend(loc="lower right")

    # Save the plot
    roc_save_path = os.path.join(save_path, 'Random_Forest_roc_curve.png')
    plt.savefig(roc_save_path, dpi=300)
    plt.show()
    plt.close()

    # Visualization SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values[1], X_train, show=False)
    # Save the plot
    shap_save_path = os.path.join(save_path, 'Random_Forest_shap_summary.png')
    plt.savefig(shap_save_path, dpi=300)
    plt.show()
    plt.close()
    # generate a dataframe to store the thresholds, fpr and tpr, only fpr less than 0.2
    df_thresholds = pd.DataFrame({
    'Threshold': thresholds,
    'FPR': fpr,
    'TPR': tpr
})  
    df_thresholds = df_thresholds[df_thresholds['FPR'] <= 0.2]
    print(df_thresholds)
    
    return model, study, df_thresholds

## Preserve aspect ratio
def preserve_aspect_ratio(img_path, max_width, max_height):
    """Calculate image size to preserve aspect ratio within bounds."""
    img = PILImage.open(img_path)
    img_width, img_height = img.size

    # Calculate ratio to fit within dimensions
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    ratio = min(width_ratio, height_ratio)

    return img_width * ratio, img_height * ratio

## Generate PDF report
def generate_pdf_report(model_name, df_thresholds, save_path):
    pdf_filename = os.path.join(save_path, f'{model_name}_report.pdf')
    roc_image_path = os.path.join(save_path, f'{model_name}_roc_curve.png')
    shap_image_path = os.path.join(save_path, f'{model_name}_shap_summary.png')
    data = [df_thresholds.columns.values.tolist()] + df_thresholds.values.tolist()
    
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    max_image_width = 500
    max_image_height = 300
    
    # Add Title
    title = Paragraph(f"Model Evaluation Report: {model_name}", styles['Heading1'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Add Roc curve
    roc_img_width, roc_img_height = preserve_aspect_ratio(roc_image_path, max_image_width, max_image_height)
    elements.append(Paragraph("ROC Curve:", styles['Heading2']))
    elements.append(Image(roc_image_path, width=roc_img_width, height=roc_img_height))
    elements.append(Spacer(1, 12))
    
    # Add SHAP
    shap_img_width, shap_img_height = preserve_aspect_ratio(shap_image_path, max_image_width, max_image_height)
    elements.append(Paragraph("SHAP Summary:", styles['Heading2']))
    elements.append(Image(shap_image_path, width=shap_img_width, height=shap_img_height))
    elements.append(Spacer(1, 12))
    
    # Add table
    elements.append(Paragraph("Thresholds Analysis:", styles['Heading2']))
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    
    doc.build(elements)

## Delete all files in a folder except for a specific one
def delete_files_except_specific_one(folder_path, file_to_keep):
    file_to_keep = f'{file_to_keep}_report.pdf'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename != file_to_keep:
#            print(f"Deleting {filename}...")
            os.remove(file_path)
        elif os.path.isdir(file_path):
            print(f"Skipping directory {filename}")