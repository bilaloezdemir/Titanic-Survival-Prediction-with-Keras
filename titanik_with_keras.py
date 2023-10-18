import pandas as pd
import zipfile
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Set Pandas display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)

# Unzip the Titanic dataset and read the 'train.csv' file
df_zip = zipfile.ZipFile("/Users/bilal/Desktop/pythonProject2/deep learnink/keras/titanic.zip")
data = pd.read_csv(df_zip.open("train.csv"))
df = data.copy()

# Drop the 'PassengerId' column
df = df.drop("PassengerId", axis=1)

# Fill missing 'Age' values with the mean 'Age' grouped by 'Survived', 'Pclass', and 'Sex'
age_mean = df.groupby(['Survived', 'Pclass', 'Sex'])['Age'].transform("mean")
df['Age'] = df["Age"].fillna(age_mean)

# Create a new DataFrame 'df_cabin' with rows containing 'Cabin' data and drop 'Cabin' from the original DataFrame
df_cabin = df.dropna(subset=["Cabin"])
df = df.drop("Cabin", axis=1)

# Fill missing 'Embarked' values with the most frequent value in the 'Embarked' column
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Extract and clean the 'Title' from the 'Name' column
df["Title"] = df["Name"].str.split(".").str.get(0)
df["Title"] = df["Title"].str.split(",").str.get(1)
df["Title"] = df["Title"].str.strip()

# Create a mapping from 'ftitle' to 'etitle' and replace 'Title' values accordingly
ftitle = ['Don', 'Dona', 'Mme', 'Ms', 'Mra', 'Mlle']
etitle = ['Mr', 'Mrs', 'Mrs', 'Mrs', 'Mrs', 'Miss']
title_mapping = dict(zip(ftitle, etitle))
df["Title"] = df["Title"].replace(title_mapping)

# Assign 'Others' to 'Title' values that are not in ['Mr', 'Mrs', 'Miss', "Master"]
df["Title"] = df["Title"].apply(lambda x: x if x in ['Mr', 'Mrs', 'Miss', "Master"] else "Others")

# Drop the 'Name' and 'Ticket' columns
df = df.drop("Name", axis=1)
df = df.drop("Ticket", axis=1)

# Encode categorical columns using LabelEncoder
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title']
le = LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

# Model
y = df["Survived"]
X = df.drop("Survived", axis=1)

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

# Create a Sequential model
model = Sequential()

# Add input layer with 64 neurons and ReLU activation function
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))

# Add dropout layer with a dropout rate of 0.2
model.add(Dropout(0.2))

# Add a hidden layer with 32 neurons and ReLU activation function
model.add(Dense(32, activation="relu"))

# Add another dropout layer with a dropout rate of 0.2
model.add(Dropout(0.2))

# Add the output layer with 1 neuron and a sigmoid activation function
model.add(Dense(1, activation="sigmoid"))

# Compile the model with Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model and store the training history
result = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0)

# Plot the training and validation accuracy over epochs
plt.figure(figsize=(25, 10))
plt.plot(result.history["accuracy"])
plt.plot(result.history["val_accuracy"])
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc='upper right')
plt.show()
