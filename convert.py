import pickle

from skl2onnx import to_onnx
from skl2onnx.common.data_types import DoubleTensorType

import onnxruntime as rt

PICKLE_FILE = 'model/give-me-some-credit_linear-svc.pickle'
ONNX_FILE = 'model/give-me-some-credit_linear-svc.onnx'

#########################################
# Load the model from the pickle

print()
print("Loading the model from", PICKLE_FILE)

with open(PICKLE_FILE, "rb") as file:
  model = pickle.load(file)

#########################################
# Saving the ONNX model

print("Saving the model to", ONNX_FILE)
print()

# save the model as ONNX
initial_type = [('double_input', DoubleTensorType([None, 10]))]
onnx_model = to_onnx(model, initial_types=initial_type)
with open(ONNX_FILE, "wb") as f:
  f.write(onnx_model.SerializeToString())
