/**
 * @brief Optical dense calculation
 * @details
 *
 * @param float 1D flat array
 */

__kernel void opticalDense(__global float *buff)
{
    int gid = get_global_id(0);
    buff[gid] = -log1p(buff[gid]);
}

/**
 * @brief From optical dence to color
 * @details
 *
 * @param float 1D flat array
 */

__kernel void toColorDense(__global float *buff)
{
    int gid = get_global_id(0);
    buff[gid] = exp(-buff[gid]);
}

/**
 * @brief Matrix multiplication: C = A * B.
 * @details http://gpgpu-computing4.blogspot.ru/2009/09/matrix-multiplication-2-opencl.html
 *
 * Second matrix *must* be wider than first. Global size be defined as (w, h) of C.
 *
 * @param float C Empty matrix, height equal to A, width equal to B
 * @param float A Source matrix, width must be same as B height
 * @param float B Source matrix, height must be same as A width
 * @param float widthA Width of array A (row count)
 * @param float widthB Width of array B (row count)
 */
__kernel void gemm(__global float* C,
                   __global const float* A,
                   __global const float* B,
                   const int widthA,
                   const int widthB)
{
    int tx = get_global_id(0);  // Width
    int ty = get_global_id(1);  // Height
    // value stores the element that is computed by the thread
    float value = 0;
    for (int k = 0; k < widthA; ++k) {
        float elementA = A[ty * widthA + k];
        float elementB = B[k * widthB + tx];
        value += elementA * elementB;
    }
    // widthA was here. Seems like error.
    C[ty * widthB + tx] = value;
}


/**
 * @brief Matrix multiplication: C = A * B.
 * @details http://gpgpu-computing4.blogspot.ru/2009/09/matrix-multiplication-2-opencl.html
 *
 * Second matrix *must* be wider than first. More clear since global size must
 * have same shape as C, defined as (h, w) of C.
 *
 * @param float C Empty matrix, height equal to A, width equal to B
 * @param float A Source matrix, width must be same as B height
 * @param float B Source matrix, height must be same as A width
 * @param float widthA Width of array A (row count)
 * @param float widthB Width of array B (row count)
 */
__kernel void gemm_slow(__global float* C,
                   __global const float* A,
                   __global const float* B,
                   const int widthA,
                   const int widthB)
{
    int ty = get_global_id(0);  // Height
    int tx = get_global_id(1);  // Width
    // value stores the element that is computed by the thread
    float value = 0;
    for (int k = 0; k < widthA; ++k) {
        float elementA = A[ty * widthA + k];
        float elementB = B[k * widthB + tx];
        value += elementA * elementB;
    }
    C[ty * widthB + tx] = value;
}
