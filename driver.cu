extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#include <stdio.h>

#define CUDA_SAFE_CALL2(call, success, failure) do { \
  cudaError err = call; \
  if (err != cudaSuccess) { \
    fprintf(stdout, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    return failure; \
  } else return success; \
  } while(0);

#define CUDA_SAFE_CALL(call) CUDA_SAFE_CALL2(call, 0, -1)

#ifdef DEVICEEMU
EXPORT int emulating = 1;
#else
EXPORT int emulating = 0;
#endif

EXPORT void *drv_alloc(int size) {
  void *data;
  CUDA_SAFE_CALL2(cudaMalloc(&data, size), data, NULL);
}

EXPORT int drv_free(void *data) {
  CUDA_SAFE_CALL(cudaFree(data));
}

EXPORT int drv_htod(void *target, void *source, int size) {
  CUDA_SAFE_CALL(cudaMemcpy(target, source, size, cudaMemcpyHostToDevice));
}

EXPORT int drv_dtoh(void *target, void *source, int size) {
  CUDA_SAFE_CALL(cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost));
}

}
