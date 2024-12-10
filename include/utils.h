#pragma once

#include <string>
#include <vector>

#define CHECK(call)                                                       \
  {                                                                       \
    const cudaError_t error = call;                                       \
    if (error != cudaSuccess) {                                           \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
    }                                                                     \
  }

#define REVERSE_INT(x)                                            \
  ((x) = (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) | \
         (((x) & 0x0000ff00) << 8) | (((x) & 0x000000ff) << 24))

std::vector<std::vector<float>> read_images(const std::string& path);
std::vector<int> read_labels(const std::string& path);
std::vector<float> read_params(const std::string& path);