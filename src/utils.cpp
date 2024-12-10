#include <fstream>
#include <iostream>
#include <string>
#include <utils.h>
#include <vector>

std::vector<std::vector<float>> read_images(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "Open file fail!" << std::endl;
    return {};
  }

  int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  file.read((char *)&num_images, sizeof(num_images));
  file.read((char *)&num_rows, sizeof(num_rows));
  file.read((char *)&num_cols, sizeof(num_cols));

  REVERSE_INT(num_images);
  REVERSE_INT(num_rows);
  REVERSE_INT(num_cols);

  int image_size = num_rows * num_cols;
  std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

  for (int i = 0; i < num_images; ++i) {
    for (int j = 0; j < image_size; ++j) {
      unsigned char pixel = 0;
      file.read((char *)&pixel, sizeof(pixel));
      images[i][j] = static_cast<float>(pixel) / 255.0f;
    }
  }

  return images;
}

std::vector<int> read_labels(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cout << "Open file fail!" << std::endl;
    return {};
  }

  int magic_number = 0, num_items = 0;
  file.read((char *)&magic_number, sizeof(magic_number));
  file.read((char *)&num_items, sizeof(num_items));

  REVERSE_INT(num_items);

  std::vector<int> labels(num_items);
  for (int i = 0; i < num_items; ++i) {
    unsigned char label = 0;
    file.read((char *)&label, sizeof(label));
    labels[i] = static_cast<int>(label);
  }

  return labels;
}

std::vector<float> read_params(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cout << "Open file fail!" << std::endl;
    return {};
  }

  std::vector<float> params;
  float param;
  while (file >> param) {
    params.push_back(param);
  }
  return params;
}