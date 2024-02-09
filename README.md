# Customer-Churn-Classifier
# Deploy a BigQuery ML Customer Churn Classifier to Vertex AI for Online Predictions

## Overview
This project focuses on deploying a customer churn classifier using BigQuery ML to Vertex AI for online predictions. We will work with a Google Analytics 4 dataset from the Flood it! mobile application to predict the likelihood of users returning to the app. The project involves various stages, including data exploration and preprocessing, model training, tuning, evaluation, and deployment to Vertex AI for online predictions.
This project has been created by referencing this [article](https://cloud.google.com/blog/topics/developers-practitioners/churn-prediction-game-developers-using-google-analytics-4-ga4-and-bigquery-ml).

## Architecture
![Architecture](https://github.com/PriyanshuAhlawat/Customer-Churn-Classifier/blob/0ac1aace9eeaf714775f9a7a7df1352e3dbb9e37/vertex-bqml-lab-architecture-diagram.png)

### BigQuery ML
BigQuery ML is a powerful tool that enables machine learning directly within BigQuery using SQL queries. It simplifies the process of training and deploying machine learning models, offering seamless integration with Google Cloud's infrastructure.

### Vertex AI
Vertex AI is Google Cloud's unified platform for machine learning and AI development. It provides scalable infrastructure and tools for deploying and managing machine learning models, including online prediction services. By deploying our model to Vertex AI, we can leverage its capabilities for real-time predictions and model monitoring.

## Objectives
Throughout this project, we aim to achieve the following objectives:

1. **Data Exploration and Preprocessing:** We will explore and preprocess the Google Analytics 4 dataset to prepare it for machine learning tasks. This involves handling missing values, feature engineering, and data transformation.

2. **Model Training:** Using BigQuery ML, we will train a customer churn classifier using the XGBoost algorithm. We will utilize the dataset to build a predictive model that can identify users at risk of churn.

3. **Model Tuning:** We will optimize the performance of our classifier by tuning hyperparameters using BigQuery ML's hyperparameter tuning features. This process involves searching for the best set of hyperparameters to improve the model's predictive accuracy.

4. **Model Evaluation:** We will evaluate the performance of our trained classifier using appropriate metrics such as accuracy, precision, recall, and F1 score. This step ensures that our model meets the desired performance criteria before deployment.

5. **Explainability Analysis:** Using BigQuery ML's Explainable AI, we will analyze and interpret our model's predictions to understand the factors influencing customer churn. This provides valuable insights into the drivers of churn and helps in making informed business decisions.

6. **Batch Prediction Generation:** We will generate batch predictions using our trained model to predict customer churn for a given dataset. This allows us to process large volumes of data efficiently and obtain churn predictions for analysis.

7. **Model Deployment to Vertex AI:** Finally, we will export our trained BigQuery ML model and deploy it to Vertex AI for online predictions. This involves setting up a prediction endpoint on Vertex AI and integrating it with our application for real-time churn prediction.

By completing these objectives, we aim to demonstrate the end-to-end process of deploying a machine learning model for customer churn prediction using BigQuery ML and Vertex AI.
