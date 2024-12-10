#include <cstdio>
#include <stdlib.h>
#include <utils.h>

#include <chrono>
#include <string>
#include <vector>

#define IDX(row, col, length) ((row) * (length) + (col))

#define INPUT_SIZE 28
#define INPUT_CHANNELS 1

#define CONV1_KERNEL_SIZE 5
#define CONV1_OUT_CHANNELS 6
#define POOL1_KERNEL_SIZE 2
#define POOL1_STRIDE 2

#define CONV2_KERNEL_SIZE 5
#define CONV2_OUT_CHANNELS 16
#define POOL2_KERNEL_SIZE 2
#define POOL2_STRIDE 2

#define FC1_OUTPUT_SIZE 120
#define FC2_OUTPUT_SIZE 84
#define FC3_OUTPUT_SIZE 10

/**
 * Input: [1, 28, 28]
 *
 * Conv1: [6, 24, 24] (5x5 kernel)
 * ReLU1: [6, 24, 24]
 * Pool1: [6, 12, 12] (2x2 max pool, stride 2)
 *
 * Conv2: [16, 8, 8] (5x5 kernel)
 * ReLU2: [16, 8, 8]
 * Pool2: [16, 4, 4] (2x2 max pool, stride 2)
 *
 * FC1: [120] (from 16*4*4)
 * ReLU FC1: [120]
 *
 * FC2: [84] (from 120)
 * ReLU FC2: [84]
 *
 * FC3: [10] (from 84)
 */

/*
in_img: 输入图片
out_img: 输出图片
ke_w: 核权重
ke_b: 核偏置
in_sz: 输入大小（边长）
out_sz: 输出大小
ke_sz: 核大小
in_chn: 输入通道数
out_chn: 输出通道数
*/
__global__ void Conv(float *in_img, float *out_img, float *ke_w, float *ke_b, int in_sz, int out_sz, int ke_sz, int in_chn, int out_chn) {
  int in_idx;
  int ke_idx;
  int out_idx;
  float res;
  for (int cur_chn = blockIdx.x; cur_chn < out_chn; cur_chn += gridDim.x) {
    for (int row = threadIdx.y; row < out_sz; row += blockDim.y) {
      for (int col = threadIdx.x; col < out_sz; col += blockDim.x) {
        // 计算 cur_chn (row, col) 的值
        out_idx = cur_chn * (out_sz * out_sz) + IDX(row, col, out_sz);
        res = 0;
        for (int chn_i = 0; chn_i < in_chn; chn_i++) {
          for (int ke_i = 0; ke_i < ke_sz; ke_i++) {
            for (int ke_j = 0; ke_j < ke_sz; ke_j++) {
              in_idx = chn_i * in_sz * in_sz + IDX(row + ke_i, col + ke_j, in_sz);
              ke_idx = cur_chn * (in_chn * ke_sz * ke_sz) + chn_i * (ke_sz * ke_sz) + IDX(ke_i, ke_j, ke_sz);
              res += in_img[in_idx] * ke_w[ke_idx];
            }
          }
        }
        out_img[out_idx] = res + ke_b[cur_chn];
      }
    }
  }
}

/*
in_img: 输入图片
out_img: 输出图片
n: 图片总像素
*/
__global__ void ReLU(float *in_img, float *out_img, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    out_img[i] = (in_img[i] <= 0) ? 0 : in_img[i];
  }
}

/*
in_img: 输入图片
out_img: 输出图片
in_sz: 输入大小
out_sz: 输出大小
ke_sz: 核大小
out_chn: 输出通道数
stride: 步长
*/
__global__ void Pool(float *in_img, float *out_img, int in_sz, int out_sz, int ke_sz, int out_chn, int stride) {
  int in_idx;
  int out_idx;
  float res;
  for (int cur_chn = blockIdx.x; cur_chn < out_chn; cur_chn += gridDim.x) {
    for (int row = threadIdx.y; row < out_sz; row += blockDim.y) {
      for (int col = threadIdx.x; col < out_sz; col += blockDim.x) {
        // 计算 cur_chn (row, col) 的值
        out_idx = cur_chn * (out_sz * out_sz) + IDX(row, col, out_sz);
        res = 0;
        for (int ke_i = 0; ke_i < ke_sz; ke_i++) {
          for (int ke_j = 0; ke_j < ke_sz; ke_j++) {
            in_idx = cur_chn * (in_sz * in_sz) + IDX(row * stride + ke_i, col * stride + ke_j, in_sz);
            res = (in_img[in_idx] > res) ? in_img[in_idx] : res;
          }
        }
        out_img[out_idx] = res;
      }
    }
  }
}

/*
in_img: 输入图片
out_img: 输出图片
fc_w: 权重
fc_b: 偏置
in_sz: 输入节点
*/
__global__ void FC(float *in_img, float *out_img, float *fc_w, float *fc_b, int in_sz) {
  int out_idx = blockIdx.x;
  for (int i = threadIdx.x; i < in_sz; i += blockDim.x) {
    float val = in_img[i] * fc_w[out_idx * in_sz + i];
    atomicAdd(&out_img[out_idx], val);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&out_img[out_idx], fc_b[out_idx]);
  }
}

int main() {
  auto images = read_images("./data/FashionMNIST/raw/t10k-images-idx3-ubyte");
  auto labels = read_labels("./data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
  auto conv1_w = read_params("./params/conv1.weight.txt");
  auto conv1_b = read_params("./params/conv1.bias.txt");
  auto conv2_w = read_params("./params/conv2.weight.txt");
  auto conv2_b = read_params("./params/conv2.bias.txt");
  auto fc1_w = read_params("./params/fc1.weight.txt");
  auto fc1_b = read_params("./params/fc1.bias.txt");
  auto fc2_w = read_params("./params/fc2.weight.txt");
  auto fc2_b = read_params("./params/fc2.bias.txt");
  auto fc3_w = read_params("./params/fc3.weight.txt");
  auto fc3_b = read_params("./params/fc3.bias.txt");

  // Conv1
  constexpr int conv1_in_sz = INPUT_SIZE, conv1_ke_sz = CONV1_KERNEL_SIZE;
  constexpr int conv1_out_sz = conv1_in_sz - conv1_ke_sz + 1;
  constexpr int conv1_in_chn = INPUT_CHANNELS, conv1_out_chn = CONV1_OUT_CHANNELS;
  float *conv1_in_img;
  float *conv1_out_img;
  float *conv1_ke_w;
  float *conv1_ke_b;
  CHECK(cudaMalloc((void **)&conv1_in_img, conv1_in_sz * conv1_in_sz * conv1_in_chn * sizeof(float)));
  CHECK(cudaMalloc((void **)&conv1_out_img, conv1_out_sz * conv1_out_sz * conv1_out_chn * sizeof(float)));
  CHECK(cudaMalloc((void **)&conv1_ke_w, conv1_ke_sz * conv1_ke_sz * conv1_in_chn * conv1_out_chn * sizeof(float)));
  CHECK(cudaMalloc((void **)&conv1_ke_b, conv1_out_chn * sizeof(float)));
  CHECK(cudaMemcpy(conv1_ke_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(conv1_ke_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 conv1_blockDim(conv1_out_sz, conv1_out_sz);
  dim3 conv1_gridDim(conv1_out_chn);

  constexpr int relu1_in_sz = conv1_out_sz;
  constexpr int relu1_in_chn = conv1_out_chn;
  float *relu1_out_img;
  CHECK(cudaMalloc((void **)&relu1_out_img, relu1_in_sz * relu1_in_sz * relu1_in_chn * sizeof(float)));
  dim3 relu1_blockDim(conv1_out_sz * conv1_out_sz);
  dim3 relu1_gridDim(conv1_out_chn);

  constexpr int pool1_in_sz = relu1_in_sz;
  constexpr int pool1_in_chn = relu1_in_chn;
  constexpr int pool1_stride = POOL1_STRIDE;
  constexpr int pool1_ke_sz = POOL1_KERNEL_SIZE;
  constexpr int pool1_out_sz = (pool1_in_sz - pool1_ke_sz) / pool1_stride + 1;
  float *pool1_out_img;
  CHECK(cudaMalloc((void **)&pool1_out_img, pool1_out_sz * pool1_out_sz * pool1_in_chn * sizeof(float)));
  dim3 pool1_blockDim(pool1_out_sz, pool1_out_sz);
  dim3 pool1_gridDim(conv1_out_chn);

  // Conv2
  constexpr int conv2_in_sz = pool1_out_sz;
  constexpr int conv2_ke_sz = CONV2_KERNEL_SIZE;
  constexpr int conv2_out_sz = conv2_in_sz - conv2_ke_sz + 1;
  constexpr int conv2_in_chn = CONV1_OUT_CHANNELS, conv2_out_chn = CONV2_OUT_CHANNELS;
  float *conv2_out_img;
  float *conv2_ke_w;
  float *conv2_ke_b;
  CHECK(cudaMalloc((void **)&conv2_out_img, conv2_out_sz * conv2_out_sz * conv2_out_chn * sizeof(float)));
  CHECK(cudaMalloc((void **)&conv2_ke_w, conv2_ke_sz * conv2_ke_sz * conv2_in_chn * conv2_out_chn * sizeof(float)));
  CHECK(cudaMalloc((void **)&conv2_ke_b, conv2_out_chn * sizeof(float)));
  CHECK(cudaMemcpy(conv2_ke_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(conv2_ke_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 conv2_blockDim(conv2_out_sz, conv2_out_sz);
  dim3 conv2_gridDim(conv2_out_chn);

  constexpr int relu2_in_sz = conv2_out_sz;
  constexpr int relu2_in_chn = conv2_out_chn;
  float *relu2_out_img;
  CHECK(cudaMalloc((void **)&relu2_out_img, relu2_in_sz * relu2_in_sz * relu2_in_chn * sizeof(float)));
  dim3 relu2_blockDim(conv2_out_sz * conv2_out_sz);
  dim3 relu2_gridDim(conv2_out_chn);

  constexpr int pool2_in_sz = relu2_in_sz;
  constexpr int pool2_in_chn = relu2_in_chn;
  constexpr int pool2_stride = POOL2_STRIDE;
  constexpr int pool2_ke_sz = POOL2_KERNEL_SIZE;
  constexpr int pool2_out_sz = (pool2_in_sz - pool2_ke_sz) / pool2_stride + 1;
  float *pool2_out_img;
  CHECK(cudaMalloc((void **)&pool2_out_img, pool2_out_sz * pool2_out_sz * pool2_in_chn * sizeof(float)));
  dim3 pool2_blockDim(pool2_out_sz, pool2_out_sz);
  dim3 pool2_gridDim(conv2_out_chn);

  // FC1
  constexpr int fc1_in_sz = pool2_out_sz * pool2_out_sz * pool2_in_chn;
  constexpr int fc1_out_sz = FC1_OUTPUT_SIZE;
  float *fc1_out_img;
  float *fc1_ke_w;
  float *fc1_ke_b;
  CHECK(cudaMalloc((void **)&fc1_out_img, fc1_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc1_ke_w, fc1_in_sz * fc1_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc1_ke_b, fc1_out_sz * sizeof(float)));
  CHECK(cudaMemcpy(fc1_ke_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(fc1_ke_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  dim3 fc1_blockDim(fc1_in_sz);
  dim3 fc1_gridDim(fc1_out_sz);

  constexpr int relu_fc1_in_sz = fc1_out_sz;
  float *relu_fc1_out_img;
  CHECK(cudaMalloc((void **)&relu_fc1_out_img, relu_fc1_in_sz * sizeof(float)));
  dim3 relu_fc1_blockDim(1);
  dim3 relu_fc1_gridDim(relu_fc1_in_sz);

  // FC2
  constexpr int fc2_in_sz = relu_fc1_in_sz;
  constexpr int fc2_out_sz = FC2_OUTPUT_SIZE;
  float *fc2_out_img;
  float *fc2_ke_w;
  float *fc2_ke_b;
  CHECK(cudaMalloc((void **)&fc2_out_img, fc2_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc2_ke_w, fc2_in_sz * fc2_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc2_ke_b, fc2_out_sz * sizeof(float)));
  CHECK(cudaMemcpy(fc2_ke_w, fc2_w.data(), fc2_in_sz * fc2_out_sz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(fc2_ke_b, fc2_b.data(), fc2_out_sz * sizeof(float), cudaMemcpyHostToDevice));
  dim3 fc2_blockDim(fc2_in_sz);
  dim3 fc2_gridDim(fc2_out_sz);

  constexpr int relu_fc2_in_sz = fc2_out_sz;
  float *relu_fc2_out_img;
  CHECK(cudaMalloc((void **)&relu_fc2_out_img, relu_fc2_in_sz * sizeof(float)));
  dim3 relu_fc2_blockDim(1);
  dim3 relu_fc2_gridDim(relu_fc2_in_sz);

  // FC3
  constexpr int fc3_in_sz = relu_fc2_in_sz;
  constexpr int fc3_out_sz = FC3_OUTPUT_SIZE;
  float *fc3_out_img;
  float *fc3_ke_w;
  float *fc3_ke_b;
  CHECK(cudaMalloc((void **)&fc3_out_img, fc3_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc3_ke_w, fc3_in_sz * fc3_out_sz * sizeof(float)));
  CHECK(cudaMalloc((void **)&fc3_ke_b, fc3_out_sz * sizeof(float)));
  CHECK(cudaMemcpy(fc3_ke_w, fc3_w.data(), fc3_in_sz * fc3_out_sz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(fc3_ke_b, fc3_b.data(), fc3_out_sz * sizeof(float), cudaMemcpyHostToDevice));
  dim3 fc3_blockDim(fc3_in_sz);
  dim3 fc3_gridDim(fc3_out_sz);

  float *host_res;
  host_res = (float *)malloc(fc3_out_sz * sizeof(float));

  int corr = 0;
  int index = 0;
  int k = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < images.size(); t++) {
    cudaMemset(conv1_in_img, 0, conv1_in_sz * conv1_in_sz * conv1_in_chn * sizeof(float));
    cudaMemset(conv1_out_img, 0, conv1_out_sz * conv1_out_sz * conv1_out_chn * sizeof(float));
    cudaMemset(relu1_out_img, 0, relu1_in_sz * relu1_in_sz * relu1_in_chn * sizeof(float));
    cudaMemset(pool1_out_img, 0, pool1_out_sz * pool1_out_sz * pool1_in_chn * sizeof(float));
    cudaMemset(conv2_out_img, 0, conv2_out_sz * conv2_out_sz * conv2_out_chn * sizeof(float));
    cudaMemset(relu2_out_img, 0, relu2_in_sz * relu2_in_sz * relu2_in_chn * sizeof(float));
    cudaMemset(pool2_out_img, 0, pool2_out_sz * pool2_out_sz * pool2_in_chn * sizeof(float));
    cudaMemset(fc1_out_img, 0, fc1_out_sz * sizeof(float));
    cudaMemset(relu_fc1_out_img, 0, relu_fc1_in_sz * sizeof(float));
    cudaMemset(fc2_out_img, 0, fc2_out_sz * sizeof(float));
    cudaMemset(relu_fc2_out_img, 0, relu_fc2_in_sz * sizeof(float));
    cudaMemset(fc3_out_img, 0, fc3_out_sz * sizeof(float));

    CHECK(cudaMemcpy(conv1_in_img, images[t].data(), images[t].size() * sizeof(float), cudaMemcpyHostToDevice));

    // Conv1
    Conv<<<conv1_gridDim, conv1_blockDim>>>(conv1_in_img, conv1_out_img, conv1_ke_w, conv1_ke_b, conv1_in_sz, conv1_out_sz, conv1_ke_sz, conv1_in_chn,
                                            conv1_out_chn);
    CHECK(cudaDeviceSynchronize());
    ReLU<<<relu1_gridDim, relu1_blockDim>>>(conv1_out_img, relu1_out_img, relu1_in_sz * relu1_in_sz * relu1_in_chn);
    CHECK(cudaDeviceSynchronize());
    Pool<<<pool1_gridDim, pool1_blockDim>>>(conv1_out_img, pool1_out_img, conv1_out_sz, pool1_out_sz, pool1_ke_sz, pool1_in_chn, pool1_stride);
    CHECK(cudaDeviceSynchronize());

    // Conv2
    Conv<<<conv2_gridDim, conv2_blockDim>>>(pool1_out_img, conv2_out_img, conv2_ke_w, conv2_ke_b, conv2_in_sz, conv2_out_sz, conv2_ke_sz,
                                            conv2_in_chn, conv2_out_chn);
    CHECK(cudaDeviceSynchronize());
    ReLU<<<relu2_gridDim, relu2_blockDim>>>(conv2_out_img, relu2_out_img, relu2_in_sz * relu2_in_sz * relu2_in_chn);
    CHECK(cudaDeviceSynchronize());
    Pool<<<pool2_gridDim, pool2_blockDim>>>(conv2_out_img, pool2_out_img, relu2_in_sz, pool2_out_sz, pool2_ke_sz, pool2_in_chn, pool2_stride);
    CHECK(cudaDeviceSynchronize());

    // FC1
    FC<<<fc1_gridDim, fc1_blockDim>>>(pool2_out_img, fc1_out_img, fc1_ke_w, fc1_ke_b, fc1_in_sz);
    CHECK(cudaDeviceSynchronize());
    ReLU<<<relu_fc1_gridDim, relu_fc1_blockDim>>>(fc1_out_img, relu_fc1_out_img, relu_fc1_in_sz);
    CHECK(cudaDeviceSynchronize());

    // FC2
    FC<<<fc2_gridDim, fc2_blockDim>>>(relu_fc1_out_img, fc2_out_img, fc2_ke_w, fc2_ke_b, fc2_in_sz);
    CHECK(cudaDeviceSynchronize());
    ReLU<<<relu_fc2_gridDim, relu_fc2_blockDim>>>(fc2_out_img, relu_fc2_out_img, relu_fc2_in_sz);
    CHECK(cudaDeviceSynchronize());

    // FC3
    FC<<<fc3_gridDim, fc3_blockDim>>>(relu_fc2_out_img, fc3_out_img, fc3_ke_w, fc3_ke_b, fc3_in_sz);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(host_res, fc3_out_img, fc3_out_sz * sizeof(float), cudaMemcpyDeviceToHost));
    index = 0;
    for (k = 0; k < 10; k++) {
      index = (host_res[k] > host_res[index]) ? k : index;
    }
    if (index == labels[t]) {
      corr++;
    }
  }
  CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  printf("Spend: %4f, Accuracy: %4f\n", diff.count(), float(corr) / float(images.size()));

  // Conv1
  cudaFree(conv1_in_img);
  cudaFree(conv1_out_img);
  cudaFree(conv1_ke_w);
  cudaFree(conv1_ke_b);
  cudaFree(relu1_out_img);
  cudaFree(pool1_out_img);

  // Conv2
  cudaFree(conv2_out_img);
  cudaFree(conv2_ke_w);
  cudaFree(conv2_ke_b);
  cudaFree(relu2_out_img);
  cudaFree(pool2_out_img);

  // FC1
  cudaFree(fc1_ke_w);
  cudaFree(fc1_ke_b);
  cudaFree(fc1_out_img);
  cudaFree(relu_fc1_out_img);

  // FC2
  cudaFree(fc2_ke_w);
  cudaFree(fc2_ke_b);
  cudaFree(fc2_out_img);
  cudaFree(relu_fc2_out_img);

  // FC3
  cudaFree(fc3_ke_w);
  cudaFree(fc3_ke_b);
  cudaFree(fc3_out_img);

  free(host_res);

  return 0;
}
