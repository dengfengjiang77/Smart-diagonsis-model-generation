# Databricks notebook source
from ipywidgets import widgets, Layout, GridBox, Label, Box, IntSlider, Dropdown, FloatText, Textarea, Button, FloatSlider, Checkbox, HBox, VBox, Accordion
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
import pandas as pd
import ast
import json
from IPython.display import clear_output, display, HTML
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming OD_Platform_method.py is in the same directory
import sys
sys.path.append('/Workspace/Users/lh641@cummins.com/Global_OD/Asset_Based_Model_platform/BackEnd')
from OD_Platform_method import *
configFilePath = "/Workspace/Users/lh641@cummins.com/Global_OD/Asset_Based_Model_platform/Config/config.json"
config = getConfig(configFilePath)




# Get the config information from the config.jason
efpa_claim_table_path = config.get("EFPA_CLAIM_TABLE_PATH")
efpa_table_path = config.get("EFPA_TABLE")
tnf_path = config.get("TNF_PATH")
fault_codes_list = config.get("FAULT_CODES")
platform_list = config.get("PLATFORM_LIST")
emission_level_list = config.get("EMISSION_LEVEL_LIST")
train_test_split = config.get("TRAIN_TEST_SPLIT")
random_seed = config.get("RANDOM_SEED")
early_stopping = config.get("EARLY_STOPPING")
max_depth = config.get("MAX_DEPTH")
max_iterations = config.get("MAX_ITERATIONS")
user_input_path = config.get("USER_INPUT_PATH")
model_type_list = config.get("MODEL_TYPE_LIST")
symptom_list = config.get("SYMPTOM_LIST")
unchanged_feature_list = config.get("UNCHANGED_FEATURE_LIST")
efpa_feature_list = config.get("EFPA_FEATURE_LIST")
efpa_temp_table_path = config.get("EFPA_DATA_PROCESS_TEMP_PATH")
efpa_agg_temp_path = config.get("EFPA_AGG_TEMP_PATH")
feature_ranking_path = config.get("FEATURE_RANKING_PATH")
model_selection_list = config.get("MODEL_SELECTION_LIST")


## global variable
user_inputs = {}
component_names_list=[]
top_10_features = []


# Widget overview, all the widgets will be posted here
wwid_widget = widgets.Text(
    value = 'NA',
    placeholder = 'Type your WWID here',
    description = 'WWID:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False
)
mt_widget = widgets.RadioButtons(
    options = model_type_list,
    value = 'Population_based_model',
    description = 'Model Type:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False
)
symptom_widget = widgets.SelectMultiple(
    options = symptom_list,
    value = [],
    description = 'Symptom:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False,
    size=10
)
fc_widget = widgets.SelectMultiple(
    options = fault_codes_list,
    value = [],
    description = 'Fault Code:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False,
    size=10
)
pf_widget = widgets.SelectMultiple(
    options = platform_list,
    value = [],
    description = 'Platform:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False,
    size=10
)


## For NA Data OOLY!!!
# year_widget = widgets.Dropdown(
#     options = ["NA"] + build_year_list,
#     value = "NA",
#     description = 'Build Year:',
#     layout = Layout(width = 'auto', height = 'auto'),
#     disabled = False
# )


## For China Data ONLY!!!
emission_level_widget = widgets.Dropdown(
    options = ["NA"] + emission_level_list,
    value = "NA",
    description = 'Emission Level:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False
)
### User input component ###
input_box = widgets.Text(
    description='Component:',
    style={'description_width': 'initial'}
)
add_com_button = widgets.Button(description="Add Component")
delete_com_button = widgets.Button(description="Delete Last Component")
com_output = widgets.Output()
def on_add_com_button_clicked(b):
    if input_box.value:
        component_names_list.append(input_box.value)
        with com_output:
            clear_output()
            print("Current Component List:", component_names_list)
            com_widget.options = component_names_list
        input_box.value = ""

def on_delete_com_button_clicked(b):
    if component_names_list:
        component_names_list.pop()
        with com_output:
            clear_output()
            print("Current Component List:", component_names_list)
            com_widget.options = component_names_list

add_com_button.on_click(on_add_com_button_clicked)
delete_com_button.on_click(on_delete_com_button_clicked)

component_submit_box = widgets.HBox([add_com_button, delete_com_button])
component_box = widgets.VBox([input_box,component_submit_box,com_output])

## Box2 widgets
com_widget = widgets.SelectMultiple(
    options=component_names_list,
    value = [],
    description = 'Component:',
    layout = Layout(width = 'auto', height = '200px'),
    disabled = False
)

search_box = widgets.Text(
    placeholder='Type to search features',
    description='Search:',
    layout=Layout(width='auto', height='auto')
)

feature_select = widgets.SelectMultiple(
    options= efpa_feature_list,
    value=[],
    description='Features:',
    layout=Layout(width='auto', height='200px'),
    disabled=False,
)
progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=6,
    description='Progress:',
    bar_style='info', # 'success', 'info', 'warning', 'danger' or ''
    orientation='horizontal'
)
# TOP features only for Box3
top_features_select = widgets.SelectMultiple(
    options= [],  # Initially null, updated in box2's click callback function
    value=[],
    description='Top Features:',
    layout=Layout(width='auto', height='200px'),
    disabled=False
)

hyperpar_choice = widgets.RadioButtons(
    options = ['Automatic(Recommend)', 'Custom(For ML Expert)'],
    value = 'Automatic(Recommend)',
    description = 'Hyperparameter:',
    layout = Layout(width = 'auto', height = 'auto'),
    disabled = False
)
train_test_split_widget = widgets.FloatSlider(
    value = train_test_split,
    min = 0,
    max = 1,
    step = 0.05,
    description = 'Train Test Split:',
    disabled = False,
    layout = Layout(width = 'auto', height = 'auto')
)
random_seed_widget = widgets.IntText(
    value = random_seed,
    description = 'Random Seed:',
    disabled = False,
    layout = Layout(width = 'auto', height = 'auto')
)
early_stopping_widget = widgets.IntText(
    value = early_stopping,
    description = 'Early Stopping:',
    disabled = False,
    layout = Layout(width = 'auto', height = 'auto')
)
max_depth_widget = widgets.IntText(
    value = max_depth,
    description = 'Max Depth:',
    disabled = False,
    layout = Layout(width = 'auto', height = 'auto')
)
max_iterations_widget = widgets.IntText(
    value = max_iterations,
    description = 'Max Iterations:',
    disabled = False,
    layout = Layout(width = 'auto', height = 'auto')
)


hyperpar_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between'
)

hyperpar_items = [
    train_test_split_widget,
    random_seed_widget,
    early_stopping_widget,
    max_depth_widget,
    max_iterations_widget
]

hyperpar = Box(hyperpar_items, layout=Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%'
))


# Box 1 set Up （Include：accordion1 set up, update acccordion function, button/box UI layout）
widgets_list1 = [wwid_widget,mt_widget]
accordion1 = widgets.Accordion(children=widgets_list1)
accordion1.set_title(0, "* WWID => Please tpye your WWID and reports will directly send to your email after model building.")
accordion1.set_title(1, "* Model Type => Please select one model type that you want to build.")

def update_accordion_options1(change):
    if change['new'] == 'Symptom_based_model':
        # new_children = [wwid_widget,mt_widget,symptom_widget,pf_widget,year_widget]
        new_children = [wwid_widget,mt_widget,symptom_widget,pf_widget,emission_level_widget,component_box]
    else:
        # new_children = [wwid_widget,mt_widget,fc_widget,pf_widget,year_widget]
        new_children = [wwid_widget,mt_widget,fc_widget,pf_widget,emission_level_widget,component_box]
    accordion1.children = new_children
    # reset the title of accordion1
    accordion1.set_title(0, "* WWID => Please tpye your WWID and reports will directly send to your email after model building.")
    accordion1.set_title(1, "* Model Type => Please select one model type that you want to build.")
    if 'Symptom_based_model' in change['new']:
        accordion1.set_title(2, "* SYMPTOM => Please select appropriate symptom for your use case.")
        accordion1.set_title(3, "* PLATFORM => Please select appropriate engine family for your use case")
        # accordion1.set_title(4, "* YEAR => Please select build year for your use case.")
        accordion1.set_title(4, "* EMISSION LEVEL => Please select emission level for your use case.")
        accordion1.set_title(5, "* SYMPTOM REALATED COMPONENT => Please type all the components relate to the symptom your selected before.")
    else:
        accordion1.set_title(2, "* FAULT CODE => Please select appropriate list of fault codes for your use case.")
        accordion1.set_title(3, "* PLATFORM => Please select appropriate engine family for your use case")
        # accordion1.set_title(4, "* YEAR => Please select build year for your use case.")
        accordion1.set_title(4, "* EMISSION LEVEL => Please select emission level for your use case.")
        accordion1.set_title(5, "* FC REALATED COMPONENT => Please type all the components relate to the faultcodes your selected before.")

button1 = widgets.Button(description="SUBMIT",style=dict(
    font_style='italic',
    font_weight='bold',
    text_color='green',
    text_decoration='underline'
))

output1 = widgets.Output()
button1.style.button_color='red'

box1_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%')
box1 = widgets.HBox(children=[accordion1,button1,output1],layout=box1_layout)


# Box 2 set Up （Include：accordion2 set up, update acccordion function, button/box UI layout）
# accordion2 set up
widgets_list2 = [com_widget, feature_select]
accordion2 = widgets.Accordion(children=widgets_list2)
accordion2.set_title(0, "* Component => Please select one component that you want to build model.")
accordion2.set_title(1, "* Feature => Please select features that you want to build model.")

# def update_accordion_options2(change):
#     # Layout
#     new_children = [VBox([com_widget]), VBox([search_box, feature_select])]
#     accordion2.children = new_children
#     # Title
#     accordion2.set_title(0, "* Component => Please select one component that you want to build model.")
#     accordion2.set_title(1, "* Feature => Please select features that you want to build model.")

button2 = widgets.Button(description="Data Aggregation",style=dict(
    font_style='italic',
    font_weight='bold',
    text_color='green',
    text_decoration='underline'
))
output2 = widgets.Output()
button2.style.button_color='red'


# special function only for search banner
def search_features(change):
    search_term = change['new'].lower()
    filtered_options = [feature for feature in efpa_feature_list if search_term in feature.lower()]
    feature_select.options = filtered_options

# relationship bonding
search_box.observe(search_features, names='value')

# layout of box2
box2_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')
box2 = VBox(children=[
    accordion2,
    progress_bar,
    button2,
    output2
], layout=box2_layout)


# Box 3 set Up （Include：accordion3 set up, update acccordion function, button/box UI layout）
# accordion3 set up
accordion3 = Accordion(children=[VBox([top_features_select]), VBox([hyperpar_choice])])
accordion3.set_title(0, "* Feature => Please select top features to refine model.")
accordion3.set_title(1, "* Hyperparameters => Default value are given, modify for model tuning.")

# update accordion3
def update_accordion_options3(change):
    if change['new'] == 'Custom(For ML Expert)':
        new_children = [top_features_select, hyperpar_choice, hyperpar]
    else:
        new_children = [top_features_select, hyperpar_choice]
    accordion3.children = new_children
    accordion3.set_title(0, "* Feature => Please select top features to refine model.")
    accordion3.set_title(1, "* Hyperparameters => Default value are given, modify for model tuning.")
    if 'Custom(For ML Expert)' in change['new']:
        accordion3.set_title(2, "Custom Hyperparameters")

# Setting up an observer for accordion3
hyperpar_choice.observe(update_accordion_options3, names='value')

button3 = widgets.Button(description="SUBMIT", style=dict(
    font_style='italic',
    font_weight='bold',
    text_color='green',
    text_decoration='underline'
))
output3 = widgets.Output()
button3.style.button_color='red'

box3_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='100%')
box3 = VBox(children=[accordion3, button3, output3], layout=box3_layout)


## Button click trigger - box1 (input wwid , etc)
def on_button1_clicked(b):   
    with output1:
        clear_output(wait=True)
        total_null_inputs = 0
        print("You have clicked submit button!")
        wwid = wwid_widget.value
        mt = mt_widget.value
        if mt == "Asset_based_model":
            if mt == "Symptom_based_model":
                symptom_fc = symptom_widget.value
            else:
                symptom_fc = fc_widget.value
            platform = pf_widget.value
            # y = year_widget.value
            emission_level = emission_level_widget.value
            if wwid == "NA":
                print("Please type your WWID!")
                total_null_inputs += 1
            if len(symptom_fc) == 0:
                print("Please select at least one symptom/fault code!")
                total_null_inputs += 1
            if len(platform) == 0:
                print("Please select a platform!")
                total_null_inputs += 1
            # if y == "NA":
            #     print("Please select build year!")
            #     total_null_inputs += 1
            if emission_level == "NA":
                print("Please select emission level!")
                total_null_inputs += 1
            if len(component_names_list) < 2:
                print("Please type at least 2 component in order to build model!")
                total_null_inputs +=1

            print("Total null inputs: ", total_null_inputs)
            if total_null_inputs == 0:
                user_inputs['wwid'] = wwid
                user_inputs['faultcode/symptom'] = list(symptom_fc)
                user_inputs['platform'] = list(platform)
                # user_inputs['year'] = y
                user_inputs['emission_level'] = [emission_level]
                b.style.button_color = 'green'
                b.description = "SUCCESS!"
                print("ALL COLUMNS ARE FILLED UP!")
            else:
                raise Exception("Please fill up all columns!")
        else:
            raise Exception("Sorry, this template only support Asset based model right now!")




# Button click trigger - box2 (feature selection, data draft aggregation)
def update_progress(increment=1):
    if progress_bar.value < progress_bar.max:
        progress_bar.value += increment

def on_button2_clicked(b):
    with output2:
        clear_output(wait=True)     
        total_null_inputs = 0
        print("You have clicked Data Aggregation button!")
        total_steps = 6
        progress_bar.max = total_steps
        progress_bar.value = 0
        com = list(com_widget.value)
        features = list(feature_select.value)
        required_columns = ['label'] + features

        if len(com) == 0:
            print("Please select at least one component!")
            total_null_inputs += 1
        if len(features) == 0:
            print("Please select at least one feature!")
            total_null_inputs += 1
        if total_null_inputs == 0:
            print("Collecting Claim Data...")
            print("Collecting TNF Data...")
            all_claim = getclaimnum(com,component_names_list,efpa_claim_table_path,tnf_path)
            print("Claim Data Read Successfully!")
            update_progress()
            print("TNF Data Read Successfully!")
            update_progress()
            print("Collecting eFPA Data...")
            raw_efpa = getefpa(unchanged_feature_list,features,efpa_table_path,user_inputs['platform'],user_inputs['emission_level'])
            print("eFPA Data Read Successfully!")
            update_progress()
            print("Filtering eFPA Data...")
            print("This may take up to 20 minutes, please wait...")
            filtered_efpa = filterefpa(all_claim,raw_efpa,user_inputs['faultcode/symptom'])
            filtered_efpa.write.mode("overwrite").option("overwriteSchema", "true").format("delta").save(efpa_temp_table_path)
            print("Filter finished!")
            update_progress()
            print("Aggregating...")

            # Ensure required columns exist in DataFrame
            missing_columns = [col for col in required_columns if col not in filtered_efpa.columns]
            if missing_columns:
                print(f"Column(s) {missing_columns} do not exist in the dataset.")
                return

            # Data preprocessing: attempt to convert strings to numeric types
            for feature in features:
                filtered_efpa = filtered_efpa.withColumn(feature, col(feature).cast('double'))

            # Remove rows with NaNs in the feature columns
            filtered_efpa = filtered_efpa.na.drop(subset=features)

            if filtered_efpa.count() == 0:
                print("The dataset is empty after preprocessing.")
                return
            
            # Data Aggregation
            aggregation_functions = {
            feature: [
                F.max(F.col(feature)).alias(f"{feature}_max"),
                F.min(F.col(feature)).alias(f"{feature}_min"),
                F.avg(F.col(feature)).alias(f"{feature}_avg"),
                F.stddev(F.col(feature)).alias(f"{feature}_std")
            ] for feature in features
            }
            aggregated_data = filtered_efpa.groupBy('ESN').agg(*(expr for sublist in aggregation_functions.values() for expr in sublist))

            unique_labels = filtered_efpa.select("ESN", "label").dropDuplicates()
            aggregated_data = aggregated_data.join(unique_labels, on="ESN", how="left")
            aggregated_data.write.mode("overwrite").option("overwriteSchema", "true").format("delta").save(efpa_agg_temp_path)
            df = aggregated_data.toPandas()
       
            # feature selection
            selected_features = []
            for feature in features:
                selected_features.extend([f"{feature}_max", f"{feature}_min",  f"{feature}_avg", f"{feature}_std"])
            
            
            X = df[selected_features]
            
            y = df['label']
            label_0 = len(df[df['label'] == 0])
            label_1 = len(df[df['label'] == 1])
            print(f"Number of Positive Data: {label_1}")
            print(f"Number of Negative Data: {label_0}")

            if len(X) != len(y):
                print("X and y lengths are not equal. Check the data preparation steps.")
                return

            if len(np.unique(y)) < 2:
                print("No positive or negative samples in y_true, cannot train the model. Please select appropriate components.")

            print(f"Received components: {com}")
            print(f"Received features: {features}")
            print("Starting feature importance ranking using XGBoost. Please ensure your data source is correct")
            update_progress()
            print("Data cooking in process, be patient....")
            
            # Train and evaluate XGBoost models
            save_path = feature_ranking_path  # Setting the result save path
            xgb_model, top_10_features, importance_sorted = ranking_xgboost(X, y, train_test_split, random_seed, early_stopping, max_depth, max_iterations, save_path)

            # Update options for top_features_select
            print("Data Aggregation Success!")
            update_progress()
            print(top_10_features)
            top_features_select.options = importance_sorted
            b.style.button_color = 'green'
            b.description = "SUCCESS!"


# Button click trigger - box3 (top10_features selection, hyperpar_choice )
def on_button3_clicked(b):
    with output3:
        clear_output(wait=True)
        total_null_inputs = 0
        selected_features = top_features_select.value
        hyperpar_choice_value = hyperpar_choice.value
        if len(selected_features) == 0:
            print("Please select at least one feature!")
            total_null_inputs += 1
        if total_null_inputs == 0:
            user_inputs['selected_features'] = list(selected_features)
            user_inputs['hyperpar_choice'] = hyperpar_choice_value
            if hyperpar_choice_value == 'Custom(For ML Expert)':
                user_inputs['train_test_split'] = train_test_split_widget.value
                user_inputs['random_seed'] = random_seed_widget.value
                user_inputs['early_stopping'] = early_stopping_widget.value
                user_inputs['max_depth'] = max_depth_widget.value
                user_inputs['max_iterations'] = max_iterations_widget.value
            with open(user_input_path, 'w') as file:
                json.dump(user_inputs, file)
            b.style.button_color = 'green'
            b.description = "SUCCESS!"
            print("You have clicked the submit button!")
            print(f"Selected features: {selected_features}")
            print(f"Hyperparameter choice: {hyperpar_choice_value}")
            if hyperpar_choice_value == 'Custom(For ML Expert)':
                custom_params = {
                    'train_test_split': train_test_split_widget.value,
                    'random_seed': random_seed_widget.value,
                    'early_stopping': early_stopping_widget.value,
                    'max_depth': max_depth_widget.value,
                    'max_iterations': max_iterations_widget.value
                }
                print(f"Custom hyperparameters: {custom_params}")
            print("ALL COLUMNS ARE FILLED UP!")
        else:
            raise Exception("Please fill up all columns!")



## Model Selection and PDF Report
model_selection_widget = widgets.RadioButtons(
    options = model_selection_list,
    value = 'XGBoost',
    description = 'Model:',
    disabled = False
)
button4 = widgets.Button(description="SUBMIT",style=dict(
    font_style='italic',
    font_weight='bold',
    text_color='green',
    text_decoration='underline'
))


accordion4 = widgets.Accordion(children=[model_selection_widget])
accordion4.set_title(0, "* WWID => Please select one model that you want to have report.")
output4 = widgets.Output()
button4.style.button_color='red'

box4_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='100%')
box4 = widgets.HBox(children=[accordion4,button4,output4],layout=box4_layout)

def on_button4_clicked(b):
    with output4:
        clear_output(wait=True)
        b.style.button_color = 'green'
        b.description = "SUCCESS!"
        model_name = model_selection_widget.value
        if model_name == 'XGBoost':
            generate_pdf_report(model_name, xgb_table, result_path)
        else:
            generate_pdf_report(model_name, rf_table, result_path)
        delete_files_except_specific_one(result_path, model_name)
        print("Congratulations! The report will be sent to your email soon!")



# Click event code
mt_widget.observe(update_accordion_options1, names='value')
button1.on_click(on_button1_clicked)
# hyperpar_choice.observe(update_accordion_options2, names='value')
button2.on_click(on_button2_clicked)
button3.on_click(on_button3_clicked)
button4.on_click(on_button4_clicked)
