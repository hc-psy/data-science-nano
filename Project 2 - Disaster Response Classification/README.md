# Disaster Response Project

In this project, disaster messages derived from [Figure Eight](https://appen.com/) was analyzed, through which I therefore built a web-app which can classify messages into 36 categories with machine-learning techniques.

![demo](demo.png)


## File Descriptions

```
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   ├── static
│   │   └── favicon.ico---------------# FAVICON FOR THE WEB APP
│   └── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE OF WEB APP
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA TO
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── models
│   └── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK
```
