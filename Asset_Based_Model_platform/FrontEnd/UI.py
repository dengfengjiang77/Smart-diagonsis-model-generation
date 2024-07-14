# Databricks notebook source
# MAGIC %md
# MAGIC # Global OD Self Service Platform

# COMMAND ----------

# MAGIC %pip install optuna reportlab

# COMMAND ----------

# MAGIC %run /Workspace/Users/wz558@cummins.com/Global_OD/Asset_Based_Model_platform/BackEnd/OD_Platform_Widgets

# COMMAND ----------

display(box1)

# COMMAND ----------

display(box2)

# COMMAND ----------

display(box3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Here is your input, please check

# COMMAND ----------

getConfig(user_input_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling

# COMMAND ----------

# MAGIC %run /Workspace/Users/wz558@cummins.com/Global_OD/Asset_Based_Model_platform/BackEnd/OD_Platform_execution_helper

# COMMAND ----------

# MAGIC %md
# MAGIC ## Please select one of the model

# COMMAND ----------

display(box4)