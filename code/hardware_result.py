import os
import re
import sys
import argparse
import time
import pdb
import random
import numpy as np
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision

# File paths for the output matrices
OUTPUT_FILE_1 = "./output1.txt"
OUTPUT_FILE_2 = "./output2.txt"
test_output_file = "./hw_test_output.txt"
golden_output_file = './estChannelGridPerfect.txt'

# Function to read a matrix from a .txt file
def read_matrix(file_path):
    try:
        matrix = np.loadtxt(file_path, dtype=np.float32)
        return matrix
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

# Read the output matrices
delimiter='\t'
with open(golden_output_file, 'r') as file:
    golden_output = []
    for line in file:
        # Split the line into values using the specified delimiter and convert to floats.
        golden_output.extend(complex(value.replace('i','j')) for value in line.strip().split(delimiter))

Readoutput_1 = read_matrix(OUTPUT_FILE_1)
Readoutput_1 = Readoutput_1.T
Readoutput_2 = read_matrix(OUTPUT_FILE_2)
Readoutput_2 = Readoutput_2.T
Readtest = read_matrix(test_output_file)
Readtest = Readtest.T


# Convert the golden output list to a PyTorch tensor.
golden_tensor = torch.tensor(golden_output, dtype=torch.cfloat)
golden_tensor=golden_tensor.real

tensor_1=torch.tensor(Readoutput_1, dtype=torch.float)
tensor_1=tensor_1.flatten()

tensor_2=torch.tensor(Readoutput_2, dtype=torch.float)
tensor_2=tensor_2.flatten()

tensor_test=torch.tensor(Readtest, dtype=torch.float)
tensor_test=tensor_test.flatten()

#Calculate the MSE Loss
loss_fn = torch.nn.MSELoss()

loss1=loss_fn(tensor_1, golden_tensor)
loss2=loss_fn(tensor_2, golden_tensor)
loss_test=loss_fn(tensor_test, golden_tensor)


# # Calculate the sum of squares
# sum_of_squares = Readoutput_1**2 + Readoutput_2**2

# sum_of_squares2 = Readoutput_1**2

# # Calculate the square root of the sum of squares
# sqrt_sum_of_squares = np.sqrt(sum_of_squares)

# sqrt_sum_of_squares2 = np.sqrt(sum_of_squares2)

# # Calculate the mean value of sqrt_sum_of_squares
# mean_value = np.mean(sqrt_sum_of_squares)
# mean_value2 = np.mean(sqrt_sum_of_squares2)

# Display the mean value
print(f"MSE Loss of the hardware result for input1 is: {loss1}")
print(f"MSE of the hardware result for input2 is: {loss2}")
print(f"MSE of the hardware result for test_input is: {loss_test}")