import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import onnxruntime as rt
# import numpy

# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType


# read in the CSV data
df = pd.read_csv('data/cs-training.csv', encoding='utf-8')

# clean up the data a bit
df = df.fillna(df.median())

# extract the features
X = df.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values

# extract the target
y = df.iloc[:, 1].values

# split out the test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# build a model
print("Building this model. This may take a minute or two...")
model = LinearSVC()
model.fit(X_train, y_train)

# make some predictions and report them
y_pred = model.predict(X_test)
print("Predictions")
print(y_pred)
print()

# evaluate
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)

print("True Negatives", tn)
print("True Positives", tp)
print("False Negatives", fn)
print("False Positives", fp)
print()
print("Accuracy", accuracy)
print()

# convert to model to ONNX
# onx = convert_sklearn(model)
# with open("linear_svc.onnx", "wb") as f:
#   f.write(onx.SerializeToString())

# # run and test the ONNX model
# sess = rt.InferenceSession("logreg_iris.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

# print("ONNX Predictions")
# print(pred_onx)
# print()
