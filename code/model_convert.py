
import torch
import onnx
import onnx2torch

import h5py


onnx_model = onnx.load("chan_est_onnx.onnx")

pytorch_model = onnx2torch.convert(onnx_model)

weights = {}

for name, param in pytorch_model.named_parameters():
    weights[name] = param.detach().cpu().numpy()

# Save the weights to an h5 file
with h5py.File('model_weights.h5', 'w') as f:
    for key, value in weights.items():
        f.create_dataset(key, data=value)



# Assuming you have your model defined as 'model'
torch.save(pytorch_model, 'cnn_model.pth') 

