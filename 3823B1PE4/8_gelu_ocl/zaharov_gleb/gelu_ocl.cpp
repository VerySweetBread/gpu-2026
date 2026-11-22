#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* kSource = R"CLC(
__kernel void gelu(__global float* data, const ulong count) {
    const size_t i = get_global_id(0);
    if (i >= count) return;

    const float x = data[i];
    // 0.5 * x * (1 + tanh(t)) = x / (1 + exp(-2 * t)).
    const float t2 = 1.5957691216057308f * (x + 0.044715f * x * x * x);
    data[i] = x / (1.0f + exp(-t2));
}
)CLC";

void Check(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": OpenCL error " +
                                 std::to_string(error));
    }
}

class OpenCLState {
public:
    explicit OpenCLState(int platform_index) : platform_index_(platform_index) {
        cl_uint platform_count = 0;
        Check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs");
        if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= platform_count) {
            throw std::invalid_argument("Invalid OpenCL platform index");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
        Check(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1,
                             &device_, nullptr), "clGetDeviceIDs (GPU 0)");

        try {
            cl_int error = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
            Check(error, "clCreateContext");
            queue_ = clCreateCommandQueue(context_, device_, 0, &error);
            Check(error, "clCreateCommandQueue");
            program_ = clCreateProgramWithSource(context_, 1, &kSource, nullptr, &error);
            Check(error, "clCreateProgramWithSource");

            error = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                std::size_t length = 0;
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                                      0, nullptr, &length);
                std::string log(length, '\0');
                if (length != 0) {
                    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,
                                          length, &log[0], nullptr);
                }
                throw std::runtime_error("clBuildProgram: " + log);
            }

            kernel_ = clCreateKernel(program_, "gelu", &error);
            Check(error, "clCreateKernel");
        } catch (...) {
            Release();
            throw;
        }
    }

    OpenCLState(const OpenCLState&) = delete;
    OpenCLState& operator=(const OpenCLState&) = delete;

    ~OpenCLState() { Release(); }

    int platform_index() const { return platform_index_; }
    cl_command_queue queue() const { return queue_; }
    cl_kernel kernel() const { return kernel_; }
    cl_mem buffer() const { return buffer_; }

    void Reserve(std::size_t bytes) {
        if (bytes <= capacity_) return;
        cl_int error = CL_SUCCESS;
        cl_mem next = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes,
                                     nullptr, &error);
        Check(error, "clCreateBuffer");
        if (buffer_) clReleaseMemObject(buffer_);
        buffer_ = next;
        capacity_ = bytes;
    }

private:
    void Release() {
        if (buffer_) clReleaseMemObject(buffer_);
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    int platform_index_;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem buffer_ = nullptr;
    std::size_t capacity_ = 0;
};

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};

    static std::mutex mutex;
    static std::unique_ptr<OpenCLState> state;
    std::lock_guard<std::mutex> lock(mutex);

    if (!state || state->platform_index() != platform) {
        state = std::make_unique<OpenCLState>(platform);
    }

    const std::size_t count = input.size();
    const std::size_t bytes = count * sizeof(float);
    state->Reserve(bytes);

    Check(clEnqueueWriteBuffer(state->queue(), state->buffer(), CL_FALSE,
                               0, bytes, input.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");

    const cl_mem buffer = state->buffer();
    const cl_ulong size = static_cast<cl_ulong>(count);
    Check(clSetKernelArg(state->kernel(), 0, sizeof(buffer), &buffer), "clSetKernelArg buffer");
    Check(clSetKernelArg(state->kernel(), 1, sizeof(size), &size), "clSetKernelArg size");

    constexpr std::size_t local = 256;
    const std::size_t global = ((count - 1) / local + 1) * local;
    Check(clEnqueueNDRangeKernel(state->queue(), state->kernel(), 1, nullptr,
                                 &global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");

    std::vector<float> result(count);
    Check(clEnqueueReadBuffer(state->queue(), buffer, CL_TRUE, 0,
                              bytes, result.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer");
    return result;
}
