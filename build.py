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
SAMPLES_FOR_PLAY = 'samples.txt'

#########################################
# Preparing the data

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

# evaluate the predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)

# report the evaluations
print()
print("scikit-learn Evaluation")
print("=======================")
print("True Negatives  :", tn)
print("True Positives  :", tp)
print("False Negatives :", fn)
print("False Positives :", fp)
print("Accuracy        :", accuracy)
print()

#########################################
# Saving the ONNX model

print("Saving the model to", MODEL_FILE)

# save the model as ONNX
onnx_model = to_onnx(model, X_train)
with open(MODEL_FILE, "wb") as f:
  f.write(onnx_model.SerializeToString())

#########################################
# Evaluating the ONNX model

# make some predictins
session = rt.InferenceSession(MODEL_FILE)
inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
label_name = outputs[0].name
prob_name = outputs[1].name

y_onnx = session.run([label_name, prob_name], {input_name: X_test})
y_pred_onnx = y_onnx[0]
y_prob_onnx = y_onnx[1]

# evaluate the predictions
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_onnx).ravel()
accuracy = accuracy_score(y_test, y_pred_onnx)

# report the evaluations
print()
print("ONNX Evaluation")
print("===============")
print("True Negatives  :", tn)
print("True Positives  :", tp)
print("False Negatives :", fn)
print("False Positives :", fp)
print("Accuracy        :", accuracy)
print()

#########################################
# Save the test data to try against RedisAI

print("Saving all the test data and predictions to", SAMPLES_FOR_PLAY)
print()

output_file = open(SAMPLES_FOR_PLAY, 'w')

for index, x in enumerate(X_test):
  output_file.write(f"Test Data:        {y_test[index]}\n")
  output_file.write(f"Predicted:        {y_pred[index]}\n")
  output_file.write(f"Confidence Score: {y_prob_onnx[index][0]}\n")
  output_file.write(f"AI.TENSORSET model:in DOUBLE 1 10 VALUES {x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {x[5]} {x[6]} {x[7]} {x[8]} {x[9]}\n")
  output_file.write("AI.MODELRUN models:gmsc:linearsvc INPUTS model:in OUTPUTS model:out:1 model:out:2\n")
  output_file.write("AI.TENSORGET model:out:1 VALUES\n")
  output_file.write("AI.TENSORGET model:out:2 VALUES\n\n\n")
