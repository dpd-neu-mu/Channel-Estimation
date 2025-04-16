#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <vitis/ai/graph_runner.hpp>
#include <glog/logging.h>

using namespace std;

// Hardcoded paths for the xmodel and input/output files
const string XMODEL_PATH = "./chesti_pt/chesti_pt.xmodel";
const string INPUT_FILE = "./test_input.txt";
const string OUTPUT_FILE_PATH = "./hw_test_output.txt";

// Function to display tensor dimensions
void display_tensor_dimensions(const vector<int>& shape) {
    LOG(INFO) << "Tensor Dimensions: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        LOG(INFO) << shape[i] << (i < shape.size() - 1 ? " x " : "");
    }
}

// Function to load a [612x14] matrix from a .txt file
vector<vector<float>> read_matrix_from_txt(const string& file_path) {
    LOG(INFO) << "Loading input matrix from file: " << file_path;
    ifstream file(file_path);
    if (!file.is_open()) {
        LOG(FATAL) << "Failed to open input file: " << file_path;
    }

    vector<vector<float>> matrix;
    string line;

    while (getline(file, line)) {
        istringstream iss(line);
        vector<float> row((istream_iterator<float>(iss)), istream_iterator<float>());
        matrix.push_back(row);
    }

    LOG(INFO) << "Successfully loaded input matrix.";
    return matrix;
}

// Function to write a [612x14] matrix to a .txt file
void write_matrix_to_txt(const string& file_path, const vector<vector<float>>& matrix) {
    LOG(INFO) << "Saving output matrix to file: " << file_path;
    ofstream file(file_path);
    if (!file.is_open()) {
        LOG(FATAL) << "Failed to open output file: " << file_path;
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << endl;
    }

    LOG(INFO) << "Output matrix saved successfully.";
}

// Function to copy input data into the tensor buffer
void set_single_batch_input(const vector<vector<float>>& input_matrix,
                            vart::TensorBuffer* tensor_buffer) {
    auto tensor = tensor_buffer->get_tensor();
    auto shape = tensor->get_shape();

    // Display tensor dimensions
    LOG(INFO) << "Declared Tensor Shape:";
    display_tensor_dimensions(shape);

    // Log allocated buffer size
    uint64_t data_in = 0u;
    size_t size_in = 0u;
    std::tie(data_in, size_in) = tensor_buffer->data({0, 0, 0, 0});
    LOG(INFO) << "Allocated Tensor Buffer Size (bytes): " << size_in;

    // Determine dimensions for one batch
    size_t rows = shape[1];
    size_t cols = shape[2];
    size_t channels = shape[3];
    size_t expected_size = rows * cols * channels * sizeof(int8_t);

    // Validate tensor dimensions (use only one batch)
    CHECK_EQ(rows, 14) << "Rows must be 14";
    CHECK_EQ(cols, 612) << "Columns must be 612";
    CHECK_EQ(channels, 1) << "Channels must be 1";
    CHECK(size_in >= expected_size) << "Tensor buffer size is insufficient for one batch";

    // Log expected and actual sizes
    LOG(INFO) << "Expected Size for One Batch (bytes): " << expected_size;

    // Cast data pointer to appropriate type
    int8_t* data = reinterpret_cast<int8_t*>(data_in);

    // Log writing process
    LOG(INFO) << "Writing data to tensor buffer...";

    // Fill the tensor: [1, 14, 612, 1]
    for (size_t i = 0; i < rows; ++i) {       // 14 rows
        for (size_t j = 0; j < cols; ++j) {   // 612 columns
            if ((i * cols + j) >= size_in) {  // Prevent overflow
                LOG(FATAL) << "Attempting to write beyond allocated buffer size";
            }
            data[i * cols + j] = static_cast<int8_t>(input_matrix[j][i]); // Transpose [612,14] -> [14,612]
        }
    }

    LOG(INFO) << "Data successfully written to tensor buffer.";
}

// Function to extract output data from the tensor buffer
vector<vector<float>> get_output_data(vart::TensorBuffer* tensor_buffer) {
    auto tensor = tensor_buffer->get_tensor();
    auto shape = tensor->get_shape();

    // Display tensor dimensions
    LOG(INFO) << "Output Tensor Shape:";
    display_tensor_dimensions(shape);

    // Ensure output tensor is 4D
    CHECK_EQ(shape.size(), 4) << "Output tensor must be 4D";

    size_t batch = shape[0];
    size_t rows = shape[1];
    size_t cols = shape[2];
    size_t channels = shape[3];

    // Ensure batch and channel dimensions are valid
    CHECK_EQ(batch, 6) << "Batch size must be 6"; // Adjust if batch size differs
    CHECK_EQ(channels, 1) << "Channels must be 1";

    // Extract the first batch (e.g., batch_index = 0)
    uint64_t data_out = 0u;
    size_t size_out = 0u;
    std::tie(data_out, size_out) = tensor_buffer->data({0, 0, 0, 0});
    const float* data = reinterpret_cast<const float*>(data_out);

    LOG(INFO) << "Reading data from tensor buffer...";
    vector<vector<float>> output_matrix(rows, vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output_matrix[i][j] = data[i * cols + j];
        }
    }

    LOG(INFO) << "Data successfully read from tensor buffer.";
    return output_matrix;
}


int main(int argc, char* argv[]) {
    // Initialize Google Logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1; // Enable logging to stderr

    // Load input matrix from the .txt file
    auto input_matrix = read_matrix_from_txt(INPUT_FILE);

    // Load model graph
    auto graph = xir::Graph::deserialize(XMODEL_PATH);
    auto attrs = xir::Attrs::create();
    auto runner = vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
    CHECK(runner != nullptr) << "Failed to create graph runner";

    // Get input and output tensor buffers
    auto input_tensor_buffers = runner->get_inputs();
    auto output_tensor_buffers = runner->get_outputs();
    CHECK_EQ(input_tensor_buffers.size(), 1) << "Model must have one input tensor";
    CHECK_EQ(output_tensor_buffers.size(), 1) << "Model must have one output tensor";

    // Preprocess: Set input tensor for the batch
    LOG(INFO) << "Processing input file: " << INPUT_FILE;
    set_single_batch_input(input_matrix, input_tensor_buffers[0]);

    // Sync input tensor buffer (limit to 1 batch)
    auto tensor = input_tensor_buffers[0]->get_tensor();
    auto shape = tensor->get_shape();

    // Calculate the size of one batch in bytes
    size_t rows = shape[1];
    size_t cols = shape[2];
    size_t channels = shape[3];
    size_t one_batch_size = rows * cols * channels * sizeof(int8_t);

    LOG(INFO) << "Synchronizing input buffer for one batch...";
    input_tensor_buffers[0]->sync_for_write(0, one_batch_size);

    // Execute the model
    auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "Graph execution failed";

    // Sync output tensor buffers
    LOG(INFO) << "Synchronizing output buffer...";
    output_tensor_buffers[0]->sync_for_read(0, output_tensor_buffers[0]->get_tensor()->get_data_size());

    // Postprocess: Extract output tensor
    auto output_matrix = get_output_data(output_tensor_buffers[0]);

    // Save final output matrix to .txt file
    write_matrix_to_txt(OUTPUT_FILE_PATH, output_matrix);

    LOG(INFO) << "Inference completed. Output saved to " << OUTPUT_FILE_PATH;
    return 0;
}