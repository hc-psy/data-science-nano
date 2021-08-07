# Disaster Response Project

In this project, disaster messages derived from [Figure Eight](https://appen.com/) was analyzed, through which I therefore built a web-app which can classify messages into 36 categories with machine-learning techniques.

## Demo

For example, when I enter the disaster related message "I am very starving. I need food and water.", my ML-multi label classifier will show us the corresponding labels.

<img src="demo.PNG" />


## File Descriptions

```
├── app
│   ├── custom_transformer.py---------# MY DISASTER RELATED TRANSFORMATION
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   ├── dis_res_ETL_pipeline.ipynb----# ETL PROCESS DETAILED VERSION
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── models
│   ├── custom_transformer.py---------# MY DISASTER RELATED TRANSFORMATION
│   └── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK
```
