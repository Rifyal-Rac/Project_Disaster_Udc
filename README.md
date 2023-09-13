# Disaster Response Pipeline Project

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Instructions](#instructions)
- [Acknowledgments](#acknowledgments)

---

## Introduction
This project is part of the Udacity Data Scientist Nanodegree Program in collaboration with Figure Eight. It involves the creation of a disaster response model capable of categorizing messages received during a disaster event in real-time. The aim is to ensure that these messages can be directed to the appropriate disaster response agencies efficiently.

The project also includes the development of a web application where disaster response workers can input messages and receive classification results.

## Project Overview
In this project, we utilize a pre-labeled dataset consisting of disaster messages to build a model. This model is designed to categorize incoming messages during a disaster, allowing for a more streamlined response from the relevant authorities. The key components of the project are:

### File Descriptions
- **app/run.py**: Python script to launch the web application.
- **app/templates**: Web dependency files (go.html & master.html) required to run the web application.
- **data/disaster_messages.csv**: Real messages sent during disaster events (provided by Figure Eight).
- **data/disaster_categories.csv**: Categories of the messages.
- **data/process_data.py**: ETL pipeline used to load, clean, extract features, and store data in an SQLite database.
- **data/ETL Pipeline Preparation.ipynb**: Jupyter Notebook used to prepare the ETL pipeline.
- **data/DisasterResponse.db**: Cleaned data stored in an SQLite database.
- **models/train_classifier.py**: ML pipeline used to load cleaned data, train the model, and save the trained model as a pickle (.pkl) file for later use.
- **models/classifier.pkl**: Pickle file containing the trained model.
- **models/ML Pipeline Preparation.ipynb**: Jupyter Notebook used to prepare the ML pipeline.

### Installation
There should be no additional libraries required for installation apart from those that come with the Anaconda distribution. The code should run smoothly with Python 3.5 and above.

### Instructions
To set up your database and model, follow these commands in the project's root directory:

1. Run the ETL pipeline to clean data and store it in the database: 
- **python .\data\process_data.py .\data\disaster_messages.csv .\data\disaster_categories.csv .\data\DisasterResponse.db**
2. Run the ML pipeline to train the classifier and save it (estimated 40 mins):
- **python .\models\train_classifier.py .\data\DisasterResponse.db .\models\classifier.pkl**
3. To run the web app, execute:
- **python .\app\run.py**
4. Access the web app by going to:
- [http://127.0.0.1:3001](http://127.0.0.1:3001)

## Acknowledgments
- Udacity for Data Scientist training program.
- Figure Eight for providing the dataset.

## Capture

1. Run data preprocessing
![image](https://github.com/Rifyal-Rac/Project_Disaster_Udc/blob/main/capture/4First%20data%20preprocessing.png)

2. Run the model pipeline
![image](https://github.com/Rifyal-Rac/Project_Disaster_Udc/blob/main/capture/2Run%20the%20model%20pipeline.png)

3. Run the webapp
![image](https://github.com/Rifyal-Rac/Project_Disaster_Udc/blob/main/capture/3Run%20webapp.png)

4. Opening Page and data distribution
![image](https://github.com/Rifyal-Rac/Project_Disaster_Udc/blob/main/capture/1Opening%20Page%20with%20training%20data%20distribution.png)

5. Sample output
![image](https://github.com/Rifyal-Rac/Project_Disaster_Udc/blob/main/capture/5Result%20output.png)





