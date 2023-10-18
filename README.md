# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using a dataset containing passenger information and survival status. We will go through the following steps:

1. Data Preprocessing
2. Data Analysis
3. Data Visualization
4. Building a Machine Learning Model
5. Model Training and Evaluation

## Data Preprocessing

- Unzip the Titanic dataset and read the 'train.csv' file.
- Remove the 'PassengerId' column, as it is not relevant to survival prediction.
- Fill missing 'Age' values with the mean 'Age' grouped by 'Survived', 'Pclass', and 'Sex'.
- Create a new DataFrame 'df_cabin' containing rows with 'Cabin' data, and drop the 'Cabin' column from the original DataFrame.
- Fill missing 'Embarked' values with the most frequent value in the 'Embarked' column.
- Extract and clean the 'Title' from the 'Name' column.
- Create a mapping from 'ftitle' to 'etitle' and replace 'Title' values accordingly.
- Assign 'Others' to 'Title' values that are not in ['Mr', 'Mrs', 'Miss', "Master"].
- Drop the 'Name' and 'Ticket' columns.
- Encode categorical columns using LabelEncoder.
- Split the data into training, validation, and test sets.

## Building a Machine Learning Model

- Create a Sequential model using Keras.
- Add input and hidden layers with ReLU activation functions.
- Add dropout layers to prevent overfitting.
- Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric.
- Train the model on the training data.
- Store the training history for evaluation.

## Data Analysis and Visualization

- Various data analysis and visualization steps are performed to understand the dataset and the relationships between features.

## Model Training and Evaluation

- A deep learning model is built and trained to predict passenger survival.
- The training and validation accuracy are visualized over epochs to evaluate model performance.

## Usage

You can clone this repository to your local machine and run the Jupyter Notebook to go through the entire project. Make sure you have the required Python libraries installed.

```bash
pip install pandas scikit-learn keras matplotlib
