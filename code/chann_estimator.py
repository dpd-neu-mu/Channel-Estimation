import torch
import scipy.io
import numpy as np

model_path = './cnn_model.pth'
input_mat_file = 'nnInput.mat'

def predict_and_convert_to_complex(model_path, input_mat_file):
    #The converted NN model - compatible with pytorch now
    channel_estimation_cnn = torch.load(model_path)
    channel_estimation_cnn.eval()
    #Load the input .mat file 
    mat = scipy.io.loadmat(input_mat_file)
    #print(mat.keys())
    input_data = mat['nnInput']
    input_tensor = torch.from_numpy(input_data).float()
    
    #print(input_tensor.size())

    #Order change (reversing) of matrix to have a python-compatible array 
    input_tensor_transposed = np.transpose(input_tensor, (3, 2, 1, 0))

    print("Prediction of the pretrained model has started...\n")
    #print(input_tensor_transposed.size())
    with torch.no_grad():
        est_channel_grid_nn = channel_estimation_cnn(input_tensor_transposed)
    #print(est_channel_grid_nn.size())
    print("Prediction finished!\n")
    # Convert results to complex numbers by merging the imag and real components
    complex_est_channel_grid_nn = est_channel_grid_nn[0, :, :, :] + 1j * est_channel_grid_nn[1, :, :, :]
    #print(complex_est_channel_grid_nn.size())
    
    #Save the prediction result to a .mat file to use later
    scipy.io.savemat('nnOutput.mat',  {'tensor_data': complex_est_channel_grid_nn})
    print("Prediction file is ready!\n")
    
    return complex_est_channel_grid_nn




