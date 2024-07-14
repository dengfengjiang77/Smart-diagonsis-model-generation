# Databricks notebook source
import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession
from IPython.display import clear_output
from ipywidgets import widgets
import sys
sys.path.append('/Workspace/Users/lh641@cummins.com/Global_OD/Asset_Based_Model_platform/BackEnd')
from OD_Platform_method import *

spark = SparkSession.builder.appName("execution").getOrCreate()
configFilePath = "/Workspace/Users/lh641@cummins.com/Global_OD/Asset_Based_Model_platform/Config/config.json"

config = getConfig(configFilePath)

## Get the user input
user_input_path = config.get("USER_INPUT_PATH")
user_inputs = getConfig(user_input_path)
user_id = user_inputs.get('wwid').lower()
component = user_inputs.get('component')
features = user_inputs.get('selected_features')
hyperpar_choice = user_inputs.get('hyperpar_choice')
if hyperpar_choice == 'Custom(For ML Expert)':
    train_test_split = user_inputs.get('train_test_split')
    random_seed = user_inputs.get('random_seed')
    early_stopping = user_inputs.get('early_stopping')
    max_depth = user_inputs.get('max_depth')
    max_iterations = user_inputs.get('max_iterations')

## Get the config
efpa_claim_table_path = config.get('EFPA_CLAIM_TABLE_PATH')
efpa_agg_temp_path = config.get("EFPA_AGG_TEMP_PATH")
user_input_path = config.get("USER_INPUT_PATH")
model_selection_list = config.get("MODEL_SELECTION_LIST")
result_path = config.get("RESULT_PATH").replace("wwid", user_id)
if not os.path.exists(result_path):
    os.mkdir(result_path)

efpa_agg = spark.read.format("delta").load(efpa_agg_temp_path)
df = efpa_agg.toPandas()
X = df[features]
y = df['label']
# y = (df['fp']==component).astype(int)

## Modelling(Automatic)
if hyperpar_choice == 'Automatic(Recommend)':
    xgb_model, xgb_study, xgb_table = train_evaluate_xgboost_with_optuna(X,y, result_path)
    rf_model, rf_study, rf_table = train_random_forest_with_optuna(X,y, result_path)
## Modelling(Cutsomized)
if hyperpar_choice == 'Custom(For ML Expert)':
    xgb_model, xgb_table = train_evaluate_xgboost(X, y, train_test_split, random_seed, early_stopping, max_depth, max_iterations, result_path)
    rf_model, rf_table = train_random_forest(X, y, train_test_split, random_seed, max_depth, max_iterations, result_path)