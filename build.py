import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import Int64TensorType

import onnxruntime as rt

TRAINING_DATA = 'data/cs-training.csv'
MODEL_FILE = 'model/give-me-some-credit_linear-svc.onnx'

#########################################
# Preparing the data

print("Loading and preparing the data from", TRAINING_DATA)
print()

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

# build a model
print("Building the model...")
print()

model = LinearSVC(dual=False, verbose=1)
model.fit(X_train, y_train)
print()
print()

#########################################
# Evaluating the model

# make some predictions
y_pred = model.predict(X_test)

# evaluate the predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)

# report the evaluations
print("Evaluation")
print("==========")
print("True Negatives  :", tn)
print("True Positives  :", tp)
print("False Negatives :", fn)
print("False Positives :", fp)
print("Accuracy        :", accuracy)
print()

#########################################
# Saving the ONNX model

print("Saving the model to", MODEL_FILE)
print()

# save the model as ONNX
onnx_model = to_onnx(model, X_train)
with open(MODEL_FILE, "wb") as f:
  f.write(onnx_model.SerializeToString())

#########################################
# Evaluating the ONNX model

# make some predictins
session = rt.InferenceSession(MODEL_FILE)
input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
y_pred_onnx = session.run([label_name], {input_name: X_test})[0]

# evaluate the predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_onnx).ravel()
accuracy = accuracy_score(y_test, y_pred_onnx)

# report the evaluations
print("ONNX Evaluation")
print("===============")
print("True Negatives  :", tn)
print("True Positives  :", tp)
print("False Negatives :", fn)
print("False Positives :", fp)
print("Accuracy        :", accuracy)
print()

#########################################
# Print some samples to try

sample = df_features.sample(n=20, random_state=0)

for index, row in sample.iterrows():
  print()
  print(row)
  print(f"(TARGET) SeriousDlqin2yrs\t\t\t{df_target[index]}")
  print()
