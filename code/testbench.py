import torch
from chann_estimator import predict_and_convert_to_complex
import scipy.io
import numpy as np

#Model, input, and output file paths
model_path = './cnn_model.pth'
input_file = 'nnInput.mat'
golden_output_file = 'goldenOutput'


#Calling the prediction function
prediction = predict_and_convert_to_complex(model_path, input_file)

#Loading the golden output from the same folder
golden_data = scipy.io.loadmat(golden_output_file)
#print(golden_data.keys())
golden_output = golden_data['estChannelGridPerfect']
#The complex output is converted to tensor format that is suitable to use
golden_tensor = torch.from_numpy(golden_output).to(torch.cfloat)

#Order change (reversing) of matrix to have a python-compatible array 
golden_output_transposed = np.transpose(golden_tensor, ( 1, 0))

#Flattening the output and the golden output
predicted_flat = prediction.flatten()
golden_flat = golden_output_transposed.flatten()

## Calculating Mean Squared Error (MSE) to test the performance of the neural network
ert = pow(np.abs(golden_flat - predicted_flat), 2)
#print(ert)

mse = np.mean(ert.numpy())

#Comparing results with the golden output.
print("Comparing against output data: \n")
if mse >= 0.1: #error threshold
    print("*****************************************************************************\n")
    print("FAIL: Output DOES NOT match the golden output\n")
    print(f"Mean Squared Error between predicted and golden output: {mse}\n")
    print("*****************************************************************************\n")
    
else:
    print("*****************************************************************************\n")
    print("PASS: The output matches the golden output!\n")
    print(f"Mean Squared Error between predicted and golden output: {mse}\n")
    print("*****************************************************************************\n")
    
