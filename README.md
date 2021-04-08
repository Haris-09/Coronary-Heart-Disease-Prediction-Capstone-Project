
# Coronary Heart Disease Prediction

This is the Capstone Project of Udacity Machine Learning with Microsoft Azure Nanodegree. In this project we were asked to use the dataset of our choice to solve the desired machine learning problem. We have to train the machine model using AutoML and Hyperdrive after that we have to deploy the best model and consume it's endpoint to get the desired result.

## Dataset

### Overview

In this project i uses the Framingham Heart Study Dataset available on[Kaggle](https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset). This dataset is actually firstly used to build the Framingham Risk Score which is a gender-specific algorithm used to estimate  the 10-year cardiovascular of an individual. The main task is to predict the risk of developing coronary heart disease in the next 10 years. This is a very popular study and the latest version of the dataset is also available on National Intitude of Health [website] https://biolincc.nhlbi.nih.gov/teaching/. an expert panel of the National Heart, Lung, and Blood Institute, which is part of the National Institutes of Health (NIH), USA is actively working on the problem and publishing the latest version actively and making it publically available for research purposes.

### Task

The task is to create a binary classification model to predict the risk of developing coronary heart disease in the next 10 years. Using the Azure ML SDK we will train the model using both AutoML and Hyperdrive. Finally we compare the models and deploy the best model as a webservice. In the case of Hyperdrive i uses logistic regression as a classification algorithm.

### Access

I downloaded the dataset from kaggle and uploaded it into github repository. For the Hyperdrive part i done some preprocessing manually on excel and uploaded it to my github repository. Finally i registered the datasets in Azure ML using TabularDatsetFactory in the notebook.

<p align="center">
  <img src="Screenshots/Datasets-Registered.PNG" height="250">
</p>

<p align="center">
  <img src="Screenshots/Datasets-Explore.PNG" height="250">
</p>

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
