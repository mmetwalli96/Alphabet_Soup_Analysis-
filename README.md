# Neural Network Charity Analysis Report

## Overview of The Analysis:

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- <b>EIN</b> and <b>NAME</b>—Identification columns.
- <b>APPLICATION_TYPE</b>—Alphabet Soup application type.
- <b>AFFILIATION</b>—Affiliated sector of industry.
- <b>CLASSIFICATION</b>—Government organization classification.
- <b>USE_CASE</b>—Use case for funding.
- <b>ORGANIZATION</b>—Organization type.
- <b>STATUS</b>—Active status.
- <b>INCOME_AMT</b>—Income classification.
- <b>SPECIAL_CONSIDERATIONS</b>—Special consideration for application.
- <b>ASK_AMT</b>—Funding amount requested.
- <b>IS_SUCCESSFUL</b>—Was the money used effectively.

the features in the provided dataset will be used to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

---

## Results:

- <b>Data Preprocessing </b>:
  - What variable(s) are considered the target(s) for your model?
  The target variable is the column "IS_SUCCESSFUL". 
  - What variable(s) are considered to be the features for your model?
  The column names in the following list `` ["CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT", "IS_SUCCESSFUL"] ``
  - What variable(s) are neither targets nor features, and should be removed from the input data?
  The column names in the following list ``["EIN", "NAME"] `` 
- <b>Compiling, Training, and Evaluating the Model</b>:
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
  I used two hidden layer and one output layer, the number of neurons is eight for the first layer and six for the second layer, as can be obsereved in this code
  <pre><code>
  # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
  number_input_features = len(application_df.columns.tolist()) - 1
  hidden_nodes_layer1 =  8
  hidden_nodes_layer2 = 6
  nn = tf.keras.models.Sequential()
  
  # First hidden layer
  nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
  
  # Second hidden layer
  nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
  
  # Output layer
  nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
  
  # Check the structure of the model
  nn.summary()
  </code></pre>
  The resean about my choice is that as first attempt to train a deep learning model and check the performance, I wanted to start with a relatively simple model that doesn't       require the utilization of lots of resources. 
  
  - Were you able to achieve the target model performance?
  I wasn't able to the the target performance, however, I created a model with a performance close to the desired tarfet score (model accuracy is <strong>72%</strong> and the target is a score above <strong>75%</strong>).
  
  - What steps did you take to try and increase model performance?
  I changed the number of the cutoff point to bin "rare" categorical variables together in a new column `` other ``, increased the number of hidden layers and the activation       function type. Despite that, the performance of the model didn't reach the desired target. 
  ---
  
## Summary
  
A deep learning model was created to predict if the funding that would be provided to an organization by Alphabet Soup will result in a successful applicant with an accuracy of <b>73%</b>. The target result wasn't achieved by the created model. An alternative model may implement Support Vector Machine (SVM) which will be easier in training, can handle non-linearity in the training data, and is more tolerant against overfitting.  
