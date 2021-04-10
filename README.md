
# Coronary Heart Disease Prediction

This is the Capstone Project of Udacity Machine Learning with Microsoft Azure Nanodegree. In this project we were asked to use the dataset of our choice to solve the desired machine learning problem. We have to train the machine model using AutoML and Hyperdrive after that we have to deploy the best model and consume it's endpoint to get the desired result.

## Dataset

### Overview

In this project i uses the Framingham Heart Study Dataset available on [Kaggle](https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset). This dataset is actually firstly used to build the Framingham Risk Score which is a gender-specific algorithm used to estimate  the 10-year cardiovascular of an individual. The main task is to predict the risk of developing coronary heart disease in the next 10 years. This is a very popular study and the latest version of the dataset is also available on National Intitude of Health [website](https://biolincc.nhlbi.nih.gov/teaching/). an expert panel of the National Heart, Lung, and Blood Institute, which is part of the National Institutes of Health (NIH), USA is actively working on the problem and publishing the latest version actively and making it publically available for research purposes. The Dataset consists of 16 Columns and 4240 Rows. 

### Task

The task is to create a binary classification model to predict the risk of developing coronary heart disease in the next 10 years. Using the Azure ML SDK we will train the model using both AutoML and Hyperdrive. Finally we compare the models and deploy the best model as a webservice. In the case of Hyperdrive i uses logistic regression as a classification algorithm.

The dataset contains 15 features that can be used for prediction. The remaining one last column "TenYearCHD" is label or Output.

| Variable | Description  |
| :-----: | :-: |
| male | Participant gender |
| age | Age at exam (years) |
| education | Highest degree |
| currentSmoker | Current cigarette smoking at exam |
| cigsPerDay| Number of cigarettes smoked each day |
| BPMeds | Use of Anti-hypertensive medication at exam |
| prevalentStroke | Prevalent Stroke |
| prevalentHyp | Prevalent Hypertensive. Subject was defined as hypertensive if treated or if second exam at which mean systolic was >=140 mmHg or mean Diastolic >=90 mmHg |
| diabetes | Diabetic according to criteria of first exam treated or first exam with casual glucose of 200 mg/dL or more |
| totChol | Serum Total Cholesterol (mg/dL) |
| sysBP | Systolic Blood Pressure (mean of last two of three measurements) (mmHg) |
| diaBP | Diastolic Blood Pressure (mean of last two of three measurements)(mmHg) |
| BMI | Body Mass Index, weight in kilograms/height meters squared |
| heartRate | Heart rate (Ventricular rate) in beats/min |
| glucose | Casual serum glucose (mg/dL) |

### Access

I downloaded the dataset from kaggle and uploaded it into github repository. For the Hyperdrive part i done some preprocessing manually on excel and uploaded it to my github repository. Finally i registered the datasets in Azure ML using TabularDatsetFactory in the notebook.

<p align="center">
  <img src="Screenshots/Datasets-Registered.PNG">
</p>

<p align="center">
  <img src="Screenshots/Datasets-Explore.PNG">
</p>

## Automated ML

I used the following AutoML Configuration Settings.

<p align="center">
  <img src="Screenshots/AutoML-Configuration.PNG">
</p>

| Configuration | Value | Explanation |
| :-----: | :-----: | :-: |
| experiment_timeout_minutes | 30 | Maximum amount of time in minutes that all iterations combined can take before the experiment terminates 30 minutes gives me better results |
| max_concurrent_iterations | 4 | To manage child runs in parallel shoul be less than maximum number of nodes of Compute Cluster |
| primary_metric | accuracy | Metric we want to optimize |
| compute_target | compute_target | Compute Cluster used for training the model |
| task | classification | As it is a binary classification problem the output can be either 0 or 1 |
| label_column_name | TenYearCHD | The output variable in this case the Coronary Heart disease in next 10 years either 0 or 1 |
| training_data | dataset | Registered training dataset Framingham Heart Disease Dataset in this case |
| enable_early_stopping | True | used to terminates early if the score is not improving |
| featurization | auto | define wheter to do featurization automatically handeled by AutoML |
| debug_log | automl_errors.log | log file to track errors in the automl process |

### Results

The best model obtained from AutoML is SparseNormalizer XGBoostClassifier with an Accuracy of 84.835 %.

Below Screenshots displays the completed AutoML Run showing the run and the Rundetails widget is completed.

<p align="center">
  <img src="Screenshots/AutoML-Run 8-Completed.PNG">
</p>

<p align="center">
  <img src="Screenshots/AutoML-RunDetails-Widget.PNG">
</p>

Below Screenshots displays the best AutoML Child Run Completed. It's Accuracy, Metrics and Id.

<p align="center">
  <img src="Screenshots/AutoML-Best Run.PNG">
</p>

<p align="center">
  <img src="Screenshots/AutoML-Best Run-Model.PNG">
</p>

<p align="center">
  <img src="Screenshots/AutoML-Best Run-Metrics.PNG">
</p>

We can improve the model by adding crossvalidation to avoid model overfitting, increase the experiment timeout minutes and enable deeplearning in AutoML configuration.

## Hyperparameter Tuning

I choosed the LogisticRegression Algorithm as it is binary Classification problem. we are trying to predict wether a patient has a potential risk of coronary heart disease in the next 10 years. The Hyperparameters of the model that are tuned using Hyperdrive are:

- --C the inverse of Regularization strength helps prevent over-fitting of the model. Values randomly used are (0.002, 0.02, 0.2, 2.0).
- max_iter the maximum number of iterations to converge the model. Values randomly used are (100, 200, 300, 500).
- Bandit Policy as an early termination policy to effectively utilize the computing resources by terminating the poor performing runs.

<p align="center">
  <img src="Screenshots/Hyperdrive-Configuration.PNG">
</p>

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
