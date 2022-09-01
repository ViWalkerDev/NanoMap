// Nanomap Copyright
// SPDX-License-Identifier: GPLv3

/// @file handlerAssert.h
///
/// @author Violet Walker
///
#ifndef NANOMAP_HANDLER_HANDLERASSERT_H_INCLUDED
#define NANOMAP_HANDLER_HANDLERASSERT_H_INCLUDED

#include <nanovdb/util/CudaDeviceBuffer.h>


#define cudaCheck(ans) \
    { \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

static inline bool gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
        return false;
    }
#endif
    return true;
}

#endif
