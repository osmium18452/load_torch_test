#include <iostream>
#include <fstream>
#include <torch/torch.h>

int main() {
#ifdef _WIN32
    const std::string data_root = R"(E:\Pycharm Projects\causal.dataset\data\swat\)";
#else
    const std::string data_root=R"(/remote-home/liuwenbo/pycproj/tsdata/data/swat/)";
#endif
    const int sensor_num = 51;
    const int train_set_row = 496800;
    const int test_set_row = 449919;
    const int label_row = 449919;
    auto train_set = (double *) malloc(train_set_row * sensor_num * sizeof(double));
    auto test_set = (double *) malloc(test_set_row * sensor_num * sizeof(double));
    auto label_set = (double *) malloc(label_row * sizeof(double));
    std::ifstream train_set_input_file(data_root + "swat_train_set.pt", std::ios::in | std::ios::binary);
    std::ifstream test_set_input_file(data_root + "swat_test_set.pt", std::ios::in | std::ios::binary);
    std::ifstream label_input_file(data_root + "swat_label_set.pt", std::ios::in | std::ios::binary);
    train_set_input_file.read((char *) train_set, train_set_row * sensor_num * sizeof(double));
    test_set_input_file.read((char *) test_set, test_set_row * sensor_num * sizeof(double));
    label_input_file.read((char *) label_set, label_row * sizeof(double));
    train_set_input_file.close();
    test_set_input_file.close();
    label_input_file.close();
    torch::Tensor train_set_tensor = torch::from_blob(train_set, {train_set_row, sensor_num}, torch::kDouble);
    torch::Tensor test_set_tensor = torch::from_blob(test_set, {test_set_row, sensor_num}, torch::kDouble);
    torch::Tensor label_set_tensor = torch::from_blob(label_set, {label_row, 1}, torch::kDouble);
    std::cout << train_set_tensor.sizes() << " " << torch::max(train_set_tensor).item<double>() << " "
              << torch::min(train_set_tensor).item<double>() << std::endl;
    std::cout << test_set_tensor.sizes() << " " << torch::max(test_set_tensor).item<double>() << " "
              << torch::min(test_set_tensor).item<double>() << std::endl;
    std::cout << label_set_tensor.sizes() << " " << torch::max(label_set_tensor).item<double>() << " "
              << torch::min(label_set_tensor).item<double>() << std::endl;
    return 0;
}
