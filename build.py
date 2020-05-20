import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pickle

TRAINING_DATA = 'data/cs-training.csv'
PICKLE_FILE = 'model/give-me-some-credit_linear-svc.pickle'
SAMPLES_FOR_PLAY = 'samples.txt'

#########################################
# Preparing the data

print()
print("Loading and preparing the data from", TRAINING_DATA)

# load from CSV
df = pd.read_csv(TRAINING_DATA, encoding='utf-8')

# clean up the data
df = df.iloc[:, 1:12]        # remove useless first columns
df = df.fillna(df.median())  # replace missing values with average

# extract the features and the target
df_features = df.iloc[:, 1:11].astype(float)
df_target = df.iloc[:, 0].astype(int)

X = df_features.values
y = df_target.values

# split out the train and test data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=0)

#########################################
# Building the model

print("Building the model...")

# build a model
model = LinearSVC(dual=False)
model.fit(X_train, y_train)

#########################################
# Evaluating the model

print("Evaluating the model...")

# make some predictions
y_pred = model.predict(X_test)
y_conf = model.decision_function(X_test)

# evaluate the predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)

# report the evaluations
print()
print("Model Evaluation")
print("=======================")
print("True Negatives  :", tn)
print("True Positives  :", tp)
print("False Negatives :", fn)
print("False Positives :", fp)
print("Accuracy        :", accuracy)
print()

#########################################
# Saving the model

print("Saving the model to", PICKLE_FILE)

with open(PICKLE_FILE, "wb") as file:
  pickle.dump(model, file)

#########################################
# Save the test data to try against RedisAI

print("Saving all the test data, predictions, and confidence scores to", SAMPLES_FOR_PLAY)
print()

output_file = open(SAMPLES_FOR_PLAY, 'w')

for index, x in enumerate(X_test):
  output_file.write(f"Expected Class:   {y_test[index]}\n")
  output_file.write(f"Predicted Class:  {y_pred[index]}\n")
  output_file.write(f"Confidence Score: {y_conf[index]}\n")
  output_file.write(f"AI.TENSORSET model:in DOUBLE 1 10 VALUES {x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {x[5]} {x[6]} {x[7]} {x[8]} {x[9]}\n")
  output_file.write("AI.MODELRUN models:gmsc:linearsvc INPUTS model:in OUTPUTS model:out:prediction model:out:score\n")
  output_file.write("AI.TENSORGET model:out:prediction VALUES\n")
  output_file.write("AI.TENSORGET model:out:score VALUES\n\n\n")
