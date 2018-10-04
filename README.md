# Credit Risk Modelling 

In this project, we are solving the problem of credit risk. We are predicting that whether the customer will pay back the loan or not. Based on the previous data, we have trained machine learning and deep learning models to do so and achived 91%(approx) accuracy on testing set. 

## How to use?

Clone the repository using `git clone git@github.com:Pirimid/credit-score-AI-models.git`. After that follow below steps. 


1. First you need to download the dataset from Kaggle. Here is the link https://www.kaggle.com/c/home-credit-default-risk.

2. Install Jupyter Notebook server along with Python or use Google Colab. Here is good medium article on how to use Google Colab https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d. 

3. Put all data files into folder named `credit_risk_data`. 

4. Run the Jupyter notebook server and open the file named `Credit Risk Modelling EDA.ipynb`. 

5. Run the notebook by `shift+enter`. You will see different kinds of plots describing hidden patterns into the data. 

6. For training different models open `Model Training for Credit Risk,ipynb`. 

## Models used. 

We have trained 3 different kind of models on the data. 
  * Logistic Regression
  * RandomForest Classifier
  * Neural Network with 3 hidden layers. 

Most of the time the accuracy on the testing set was around 91% for all the 3 models. Accuracy can be increased by tunning the models properly, but for now we are using them as defualt. 
