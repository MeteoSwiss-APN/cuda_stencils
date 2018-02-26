#include "cuda_runtime.h"
#include "../tools.h"
#include "../defs.h"
#include "udefs.hpp"
#include "helpers.hpp"
#include "converter.hpp"
#include "umesh.hpp"

template <typename T>
__global__ void copy(T const *__restrict__ a, T *__restrict__ b,
                     const unsigned int init_offset, const unsigned int cstride,
                     const unsigned int jstride, const unsigned int kstride,
                     const unsigned int ksize, const unsigned int isize,
                     const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  if (threadIdx.x < bsi && threadIdx.y < bsj) {

    for (int k = 0; k < ksize; ++k) {
      if (threadIdx.x < bsi && threadIdx.y < bsj) {
        b[idx] = a[idx];
        b[idx + cstride] = a[idx + cstride];
      }
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void copy_mesh(T const *__restrict__ a, T *__restrict__ b,
                          const unsigned int init_offset,
                          const unsigned int kstride, const unsigned int ksize,
                          const unsigned int mesh_size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < mesh_size) {
    for (int k = 0; k < ksize; ++k) {
      b[idx] = a[idx];
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void on_cells_ilp(T const *__restrict__ a, T *__restrict__ b,
                         const unsigned int init_offset,
                         const unsigned int cstride, const unsigned int jstride,
                         const unsigned int kstride, const unsigned int ksize,
                         const unsigned int isize, const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  if (threadIdx.x < bsi && threadIdx.y < bsj) {
    for (int k = 0; k < ksize; ++k) {
      // color 0
      b[idx] = a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride];
      // color 1
      b[idx + cstride] = a[idx] + a[idx + 1] + a[idx + cstride * 2];
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void on_cells(T const *__restrict__ a, T *__restrict__ b,
                         const unsigned int init_offset,
                         const unsigned int cstride, const unsigned int jstride,
                         const unsigned int kstride, const unsigned int ksize,
                         const unsigned int isize, const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = (blockIdx.y * blockDim.y + threadIdx.y)/2;
  const int c = (blockIdx.y * blockDim.y + threadIdx.y)%2;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY*2
                               : (jsize - blockIdx.y * BLOCKSIZEY)*2;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  if (threadIdx.x < bsi && threadIdx.y < bsj) {
    if(c==0) {
    for (int k = 0; k < ksize; ++k) {
      // color 0
      b[idx] = a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride];
      idx += kstride;

    }
    }
    else {
    for (int k = 0; k < ksize; ++k) {
      // color 1
      b[idx + cstride] = a[idx] + a[idx + 1] + a[idx + cstride * 2];
      idx += kstride;
    }
    }
  }
}



template <typename T>
__global__ void
on_cells_mesh(T const *__restrict__ a, T *__restrict__ b,
              const unsigned int kstride, const unsigned int ksize,
              const unsigned int mesh_size, sneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];
  const size_t shared_stride = blockDim.x;
  const size_t stride =
      table.isize() * table.jsize() * num_colors(table.nloc());

  tab[threadIdx.x + shared_stride * 0] = table.raw_data(idx2 + 0 * stride);
  tab[threadIdx.x + shared_stride * 1] = table.raw_data(idx2 + 1 * stride);
  tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

  __syncthreads();

  if (idx2 < mesh_size) {
    for (int k = 0; k < (int)ksize; ++k) {
      b[idx] = a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
               a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
               a[k * kstride + tab[threadIdx.x + 2 * shared_stride]];
      /*a[k*kstride+table.raw_data(idx2 + 0 * stride)] +
                  a[k*kstride+table.raw_data(idx2 + 1 * stride)] +
                  a[k*kstride+table.raw_data(idx2 + 2 * stride)];
   */
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void
on_cells_umesh(T const *__restrict__ a, T *__restrict__ b,
               const unsigned int kstride, const unsigned int ksize,
               const unsigned int mesh_size, uneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];
  const size_t shared_stride = blockDim.x;
  const size_t stride = table.totald_size();

  tab[threadIdx.x + shared_stride * 0] = table.raw_data(idx2 + 0 * stride);
  tab[threadIdx.x + shared_stride * 1] = table.raw_data(idx2 + 1 * stride);
  tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

  __syncthreads();

  if (idx2 < mesh_size) {
    for (int k = 0; k < (int)ksize; ++k) {

      b[idx] = a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
               a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
               a[k * kstride + tab[threadIdx.x + 2 * shared_stride]];
      /*a[k*kstride+table.raw_data(idx2 + 0 * stride)] +
                  a[k*kstride+table.raw_data(idx2 + 1 * stride)] +
                  a[k*kstride+table.raw_data(idx2 + 2 * stride)];
   */
      idx += kstride;
    }
  }
}

template <typename T>
__global__ void
complex_on_cells_ori(T const *__restrict__ a, T *__restrict__ b,
                 T const *__restrict__ c, T *__restrict__ d,
                 T const *__restrict__ fac1, T const *__restrict__ fac2,
                 const unsigned int init_offset, const unsigned int cstride,
                 const unsigned int jstride, const unsigned int kstride,
                 const unsigned int ksize, const unsigned int isize,
                 const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);

  // equations are
  // b = on_cell(a) * fac1 + on_cell(c)
  // d = on_cell(b) * fac2(k+1) + on_cell(a)
  if (threadIdx.x < bsi && threadIdx.y < bsj) {
    for (int k = 0; k < ksize - 1; ++k) {
      // color 0
      b[idx] = (a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride]) *
                   fac1[idx] +
               (c[idx + cstride - 1] + c[idx + cstride] + c[idx - cstride]);
      // color 1
      b[idx + cstride] =
          (a[idx] + a[idx + 1] + a[idx + cstride * 2]) * fac1[idx + cstride] +
          (c[idx] + c[idx + 1] + c[idx + cstride * 2]);

      __syncthreads();
      // color 0
      d[idx] = (b[idx + cstride - 1] + b[idx + cstride] + b[idx - cstride]) *
                   (fac2[idx + kstride] - fac2[idx] + 0.1) +
               (a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride]);
      // color 1
      d[idx + cstride] =
          (b[idx] + b[idx + 1] + b[idx + cstride * 2]) *
              (fac2[idx + cstride + kstride] - fac2[idx + cstride] + 0.1) +
          (a[idx] + a[idx + 1] + a[idx + cstride * 2]);

      idx += kstride;
    }

    // ksize-1
    int k = ksize - 1;

    // color 0
    b[idx] = (a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride]) *
                 fac1[idx] +
             (c[idx + cstride - 1] + c[idx + cstride] + c[idx - cstride]);
    // color 1
    b[idx + cstride] =
        (a[idx] + a[idx + 1] + a[idx + cstride * 2]) * fac1[idx + cstride] +
        (c[idx] + c[idx + 1] + c[idx + cstride * 2]);

    __syncthreads();
    // color 0
    d[idx] = (b[idx + cstride - 1] + b[idx + cstride] + b[idx - cstride]) *
                 fac2[idx] +
             (a[idx + cstride - 1] + a[idx + cstride] + a[idx - cstride]);
    // color 1
    d[idx + cstride] =
        (b[idx] + b[idx + 1] + b[idx + cstride * 2]) * fac2[idx + cstride] +
        (a[idx] + a[idx + 1] + a[idx + cstride * 2]);
  }
}

template <typename T>
__global__ void
complex_on_cells_mesh_ori(T const *__restrict__ a, T *__restrict__ b,
                      T *const __restrict__ c, T *__restrict__ d,
                      T *const __restrict__ fac1, T *const __restrict__ fac2,
                      const unsigned int kstride, const unsigned int ksize,
                      const unsigned int mesh_size, sneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];
  const size_t shared_stride = blockDim.x;
  const size_t stride =
      table.isize() * table.jsize() * num_colors(table.nloc());

  tab[threadIdx.x + shared_stride * 0] = table.raw_data(idx2 + 0 * stride);
  tab[threadIdx.x + shared_stride * 1] = table.raw_data(idx2 + 1 * stride);
  tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

  __syncthreads();

  if (idx2 < mesh_size) {
    for (int k = 0; k < (int)ksize - 1; ++k) {
      b[idx] = (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                   fac1[idx] +
               (c[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                c[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                c[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);

      __syncthreads();

      d[idx] = (b[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                b[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                b[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                   (fac2[idx + kstride] - fac2[idx] + 0.1) +
               (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);

      idx += kstride;
    }

    int k = ksize - 1;
    b[idx] = (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
              a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
              a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                 fac1[idx] +
             (c[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
              c[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
              c[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);

    __syncthreads();

    d[idx] = (b[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
              b[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
              b[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                 fac2[idx] +
             (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
              a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
              a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);
  }
}

template <typename T>
__global__ void
complex_on_cells(T const *__restrict__ a, T *__restrict__ b,
                 T const *__restrict__ c, T *__restrict__ d,
                 T const *__restrict__ fac1, T const *__restrict__ fac2,
                 const unsigned int init_offset, const unsigned int cstride,
                 const unsigned int jstride, const unsigned int kstride,
                 const unsigned int ksize, const unsigned int isize,
                 const unsigned int jsize) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
                               ? BLOCKSIZEX
                               : isize - blockIdx.x * BLOCKSIZEX;
  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
                               ? BLOCKSIZEY
                               : jsize - blockIdx.y * BLOCKSIZEY;

  unsigned idx = uindex(i, 0, j, cstride, jstride, init_offset);
  
  // equations are
  // b = on_cell(a) * fac1 + on_cell(c)
  // d = on_cell(b) * fac2(k+1) + on_cell(a)
  if (threadIdx.x < bsi && threadIdx.y < bsj) {
    int off1 = idx + cstride - 1;
    int off2 = idx + cstride;
    int off3 = idx - cstride;
    for (int k = 0; k < ksize - 1; ++k) {
      // color 0
      T oncella = (a[off1] + a[off2] + a[off3]) ; 
      T oncellc = (c[off1] + c[off2] + c[off3]);
      T bidx = oncella * fac1[idx] + oncellc;
      b[idx] = bidx;
         
      // color 1
      T oncella2 = (a[idx] + a[idx + 1] + a[off2 * 2]);
      T oncellc2 = (c[idx] + c[idx + 1] + c[off2 * 2]);
      T boff2 = oncella2 * fac1[off2] + oncellc2;
      b[off2] = boff2;

      __syncthreads();
      // color 0
      boff2 += b[off1] + b[off3];
      d[idx] = boff2  *  (fac2[idx + kstride] - fac2[idx] + 0.1) + oncella;
      // color 1
      bidx += b[idx +1] + b[off2 * 2];
      d[off2] = bidx * (fac2[off2 + kstride] - fac2[off2] + 0.1) + oncella2;

      idx += kstride;
    }
    // color 0
    T oncella = (a[off1] + a[off2] + a[off3]);
    b[idx] =  oncella * fac1[idx] +
             (c[off1] + c[off2] + c[off3]);
    // color 1
    T oncella2 = (a[idx] + a[idx + 1] + a[off2 * 2]);
    b[off2] =     oncella2 * fac1[off2] +
        (c[idx] + c[idx + 1] + c[off2 * 2]);

    __syncthreads();
    // color 0
    d[idx] = (b[off1] + b[off2] + b[off3]) *
                 fac2[idx] +
             oncella;
    // color 1
    d[off2] =
        (b[idx] + b[idx + 1] + b[off2 * 2]) * fac2[off2] +
        oncella2;
  }
}

template <typename T>
__global__ void
complex_on_cells_mesh(T const *__restrict__ a, T *__restrict__ b,
                      T *const __restrict__ c, T *__restrict__ d,
                      T *const __restrict__ fac1, T *const __restrict__ fac2,
                      const unsigned int kstride, const unsigned int ksize,
                      const unsigned int mesh_size, sneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];
  const size_t shared_stride = blockDim.x;
  const size_t stride =
      table.isize() * table.jsize() * num_colors(table.nloc());

  tab[threadIdx.x ] = table.raw_data(idx2 );
  tab[threadIdx.x + shared_stride ] = table.raw_data(idx2 +  stride);
  tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

  __syncthreads();

  if (idx2 < mesh_size) {

    int tab2 = threadIdx.x + shared_stride;
    int tab3 = threadIdx.x + shared_stride + shared_stride;
    for (int k = 0; k < (int)ksize - 1; ++k) {
      int off = k*kstride;      
      T oncella = (a[off + tab[threadIdx.x ]] +
		   a[off + tab[tab2]] +
		   a[off + tab[tab3]]);

      T oncellc = (c[off + tab[threadIdx.x ]] +
                   c[off + tab[tab2]] +
                   c[off + tab[tab3]]);

      b[idx] =  oncella * fac1[idx] + oncellc;

      __syncthreads();

      
      oncellc = (b[off + tab[threadIdx.x ]] +
                 b[off + tab[tab2]] +
                 b[off + tab[tab3]]);
      d[idx] =  oncellc * (fac2[idx + kstride] - fac2[idx] + 0.1) + oncella;
	
      idx += kstride;
    }

    int off = (ksize-1) * kstride;
    T oncella = (a[off + tab[threadIdx.x ]] +
		   a[off + tab[tab2]] +
		   a[off + tab[tab3]]);
    
    b[idx] = oncella * fac1[idx] +
             (c[off + tab[threadIdx.x ]] +
              c[off + tab[tab2]] +
              c[off + tab[tab3]]);

    __syncthreads();

    d[idx] = (b[off + tab[threadIdx.x ]] +
              b[off + tab[tab2]] +
              b[off + tab[tab3]]) *
      fac2[idx] + oncella;
  }
}

template <typename T>
__global__ void
complex_on_cells_umesh(T const *__restrict__ a, T *__restrict__ b,
                       T *const __restrict__ c, T *__restrict__ d,
                       T *const __restrict__ fac1, T *const __restrict__ fac2,
                       const unsigned int kstride, const unsigned int ksize,
                       const unsigned int mesh_size, uneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  extern __shared__ size_t tab[];
  const size_t shared_stride = blockDim.x;
  const size_t stride = table.totald_size();

  tab[threadIdx.x + shared_stride * 0] = table.raw_data(idx2 + 0 * stride);
  tab[threadIdx.x + shared_stride * 1] = table.raw_data(idx2 + 1 * stride);
  tab[threadIdx.x + shared_stride * 2] = table.raw_data(idx2 + 2 * stride);

  __syncthreads();

  if (idx2 < mesh_size) {
      for (int k = 0; k < (int)ksize - 1; ++k) {
        b[idx] = (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                  a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                  a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                     fac1[idx]
 +
                 (c[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                  c[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                  c[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);
        __syncthreads();
        d[idx] = (b[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                  b[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                  b[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                     (fac2[idx + kstride] - fac2[idx] + 0.1) +
                 (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                  a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                  a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);
        idx += kstride;
      }

      int k = ksize - 1;
      b[idx] = (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                   fac1[idx] +
               (c[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                c[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                c[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);

      __syncthreads();

      d[idx] = (b[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                b[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                b[k * kstride + tab[threadIdx.x + 2 * shared_stride]]) *
                   fac2[idx] +
               (a[k * kstride + tab[threadIdx.x + 0 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 1 * shared_stride]] +
                a[k * kstride + tab[threadIdx.x + 2 * shared_stride]]);
    }
}

/*
template <typename T>
__global__ void
on_cells_umesh(T const *__restrict__ a, T *__restrict__ b,
               const unsigned int init_offset, const unsigned int kstride,
               const unsigned int ksize, const unsigned int mesh_size,
               uneighbours_table table) {
  const unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idx = idx2;

  if (idx < mesh_size) {
    for (int k = 0; k < ksize; ++k) {
      b[idx] = a[idx + table(idx2, 0)] + a[idx + table(idx2, 1)] +
               a[idx + table(idx2, 2)];
      idx += kstride;
    }
  }
}
*/
// template <typename T>
//__global__ void
//    FNNAME(copyi1)(T *__restrict__ a, T *__restrict__ b,
//                   const unsigned int init_offset, const unsigned int jstride,
//                   const unsigned int kstride, const unsigned int ksize,
//                   const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumi1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgi)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - 1]) + LOAD(a[idx + 1]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumj1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + jstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgj)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - jstride]) + LOAD(a[idx + jstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(sumk1)(T *__restrict__ a, T *__restrict__ b,
//                  const unsigned int init_offset, const unsigned int jstride,
//                  const unsigned int kstride, const unsigned int ksize,
//                  const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + kstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(avgk)(T *__restrict__ a, T *__restrict__ b,
//                 const unsigned int init_offset, const unsigned int jstride,
//                 const unsigned int kstride, const unsigned int ksize,
//                 const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx - kstride]) + LOAD(a[idx + kstride]);
//    }
//    idx += kstride;
//  }
//}

// template <typename T>
//__global__ void
//    FNNAME(lap)(T *__restrict__ a, T *__restrict__ b,
//                const unsigned int init_offset, const unsigned int jstride,
//                const unsigned int kstride, const unsigned int ksize,
//                const unsigned int isize, const unsigned int jsize) {
//  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

//  const unsigned int bsi = (blockIdx.x + 1) * BLOCKSIZEX < isize
//                               ? BLOCKSIZEX
//                               : isize - blockIdx.x * BLOCKSIZEX;
//  const unsigned int bsj = (blockIdx.y + 1) * BLOCKSIZEY < jsize
//                               ? BLOCKSIZEY
//                               : jsize - blockIdx.y * BLOCKSIZEY;

//  unsigned idx = index(i, j, jstride, init_offset);
//  for (int k = 0; k < ksize; ++k) {
//    if (threadIdx.x < bsi && threadIdx.y < bsj) {
//      b[idx] = LOAD(a[idx]) + LOAD(a[idx + 1]) + LOAD(a[idx - 1]) +
//               LOAD(a[idx + jstride]) + LOAD(a[idx - jstride]);
//    }
//    idx += kstride;
//  }
//}

template <typename T>
void verify_on_cells(T *b_cell, T *a_cell, size_t isize, size_t jsize,
                     size_t ksize, size_t cstride_cell, size_t jstride_cell,
                     size_t kstride_cell, size_t init_offset) {
  for (unsigned int i = 0; i < isize; ++i) {
    for (unsigned int j = 0; j < jsize; ++j) {
      for (unsigned int k = 0; k < ksize; ++k) {
        if ((b_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                            kstride_cell, init_offset)] !=
             a_cell[uindex3(i - 1, 1, j, k, cstride_cell, jstride_cell,
                            kstride_cell, init_offset)] +
                 a_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] +
                 a_cell[uindex3(i, 1, j - 1, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)]))
          printf("Error c0 in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                 b_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)],
                 a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)]);

        if ((b_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                            kstride_cell, init_offset)] !=
             a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                            kstride_cell, init_offset)] +
                 a_cell[uindex3(i + 1, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] +
                 a_cell[uindex3(i, 0, j + 1, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)])) {
          printf("Error c1 in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                 b_cell[uindex3(i, 1, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)],
                 a_cell[uindex3(i, 0, j, k, cstride_cell, jstride_cell,
                                kstride_cell, init_offset)] +
                     a_cell[uindex3(i + 1, 0, j, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)] +
                     a_cell[uindex3(i, 0, j + 1, k, cstride_cell, jstride_cell,
                                    kstride_cell, init_offset)]);
        }
      }
    }
  }
}

template <typename T>
void verify_on_ucells(T *b_ucell, T *a_ucell, size_t ksize, mesh mesh_) {
  auto &table = mesh_.get_elements(location::cell).table(location::cell);
  const size_t kstride = mesh_.totald_size() * num_colors(location::cell);
  for (size_t k = 0; k < ksize; ++k) {
    for (size_t idx = 0; idx < mesh_.compd_size(); ++idx) {
      if (b_ucell[idx + k * kstride] != (a_ucell[table(idx, 0) + k * kstride] +
                                         a_ucell[table(idx, 1) + k * kstride] +
                                         a_ucell[table(idx, 2) + k * kstride]))
        std::cout << "ERROR " << idx << " " << b_ucell[idx + k * kstride]
                  << " ; " << (a_ucell[table(idx, 0) + k * kstride] +
                               a_ucell[table(idx, 1) + k * kstride] +
                               a_ucell[table(idx, 2) + k * kstride])
                  << "  " << a_ucell[table(idx, 0) + kstride] << " "
                  << table(idx, 0) << " " << kstride << " " << a_ucell[17424]
                  << std::endl;
    }
  }
}

template <typename T>
void verify_on_ucells(T *b_ucell, T *a_ucell, size_t ksize, umesh umesh_) {
  auto &table = umesh_.get_elements(location::cell).table(location::cell);
  const size_t kstride = umesh_.totald_size() * num_colors(location::cell);
  for (size_t k = 0; k < ksize; ++k) {
    for (size_t idx = 0; idx < umesh_.compd_size(); ++idx) {
      if (b_ucell[idx + k * kstride] != (a_ucell[table(idx, 0) + k * kstride] +
                                         a_ucell[table(idx, 1) + k * kstride] +
                                         a_ucell[table(idx, 2) + k * kstride]))
        std::cout << "ERROR " << idx << " " << b_ucell[idx + k * kstride]
                  << " ; " << (a_ucell[table(idx, 0) + k * kstride] +
                               a_ucell[table(idx, 1) + k * kstride] +
                               a_ucell[table(idx, 2) + k * kstride])
                  << "  " << a_ucell[table(idx, 0) + kstride] << " "
                  << table(idx, 0) << " " << kstride << " " << a_ucell[17424]
                  << std::endl;
    }
  }
}


template<typename T>
void init_data(T* a_data, T* c_data, T* fac1_data, T* fac2_data, location loc, size_t isize, size_t jsize, size_t ksize, size_t cstride, size_t jstride, size_t kstride, size_t init_offset) {
  for (unsigned int i = 0; i < isize; ++i) {
    for (unsigned int c = 0; c < num_colors(loc); ++c) {
      for (unsigned int j = 0; j < jsize; ++j) {
        for (unsigned int k = 0; k < ksize; ++k) {
          a_data[uindex3(i, c, j, k, cstride, jstride, kstride,
                         init_offset)] =
              uindex3(i, c, j, k, cstride, jstride, kstride,
                      init_offset);
          c_data[uindex3(i, c, j, k, cstride, jstride, kstride,
                         init_offset)] =
              uindex3(i, c, j, k, cstride, jstride, kstride,
                      init_offset) -
              cstride / 1.5;
          fac1_data[uindex3(i, c, j, k, cstride, jstride,
                            kstride, init_offset)] =
              uindex3(i, c, j, k, cstride, jstride, kstride,
                      init_offset) -
              cstride / 1.5 + i / (double)isize;

          fac2_data[uindex3(i, c, j, k, cstride, jstride,
                            kstride, init_offset)] =
              uindex3(i, c, j, k, cstride, jstride, kstride,
                      init_offset) -
              cstride / 1.5 + j / (double)jsize;
        }
      }
    }
  }

}

template<typename T>
void reset_data(T* data, location loc, size_t isize, size_t jsize, size_t ksize, size_t cstride, size_t jstride, size_t kstride, size_t init_offset) {
  for (unsigned int i = 0; i < isize; ++i) {
    for (unsigned int c = 0; c < num_colors(loc); ++c) {
      for (unsigned int j = 0; j < jsize; ++j) {
        for (unsigned int k = 0; k < ksize; ++k) {
          data[uindex3(i, c, j, k, cstride, jstride, kstride,
                         init_offset)] = -1;
        }
      }
    }
  }

}

template<typename T>
void reset_udata(T* data, size_t mesh_size, location loc)
{
  for (size_t i = 0; i < mesh_size * num_colors(loc);
       ++i) {
    data[i] = -1;
  }
}

template <typename T>
void launch(std::vector<double> &timings, mesh &mesh_, const unsigned int isize,
            const unsigned int jsize, const unsigned ksize,
            const unsigned tsteps, const unsigned warmup_step) {

  const unsigned int halo = 2;
  const unsigned int alignment = 32;
  const unsigned int right_padding = alignment - (isize + halo * 2) % alignment;
  const unsigned int first_padding = alignment - halo;
  const unsigned int cstride_cell = (isize + halo * 2 + right_padding);
  const unsigned int jstride_cell = cstride_cell * num_colors(location::cell);
  const unsigned int kstride_cell = jstride_cell * (jsize + halo * 2);
  const unsigned int total_size_cell =
      first_padding + kstride_cell * (ksize + halo * 2);

  const unsigned int init_offset =
      initial_offset(first_padding, halo, jstride_cell, kstride_cell);

  T *a_cell, *a_ucell;
  T *b_cell, *b_ucell;
  T *c_cell, *c_ucell;
  T *d_cell, *d_ucell;
  T *fac1_cell, *fac1_ucell;
  T *fac2_cell, *fac2_ucell;

  cudaMallocManaged(&a_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&b_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&c_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&d_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&fac1_cell, sizeof(T) * total_size_cell);
  cudaMallocManaged(&fac2_cell, sizeof(T) * total_size_cell);

  std::cout << " SIZ " << mesh_.totald_size() * ksize << std::endl;
  cudaMallocManaged(&a_ucell, sizeof(T) * mesh_.totald_size() *
                                  num_colors(location::cell) * ksize);
  cudaMallocManaged(&b_ucell, sizeof(T) * mesh_.totald_size() *
                                  num_colors(location::cell) * ksize);
  cudaMallocManaged(&c_ucell, sizeof(T) * mesh_.totald_size() *
                                  num_colors(location::cell) * ksize);
  cudaMallocManaged(&d_ucell, sizeof(T) * mesh_.totald_size() *
                                  num_colors(location::cell) * ksize);
  cudaMallocManaged(&fac1_ucell, sizeof(T) * mesh_.totald_size() *
                                     num_colors(location::cell) * ksize);
  cudaMallocManaged(&fac2_ucell, sizeof(T) * mesh_.totald_size() *
                                     num_colors(location::cell) * ksize);

  init_data<T>(a_cell, c_cell, fac1_cell, fac2_cell, location::cell, isize, jsize, ksize, cstride_cell, jstride_cell, kstride_cell, init_offset);

  reset_data<T>(b_cell, location::cell, isize, jsize, ksize, cstride_cell, jstride_cell, kstride_cell, init_offset);
  reset_data<T>(d_cell, location::cell, isize, jsize, ksize, cstride_cell, jstride_cell, kstride_cell, init_offset);

  for (size_t i = 0; i < mesh_.totald_size() * num_colors(location::cell);
       ++i) {
    a_ucell[i] = i;
    c_ucell[i] = i - cstride_cell / 1.5;
    fac1_ucell[i] = i - cstride_cell / 1.5 + i / (double)isize;
    fac2_ucell[i] = i - cstride_cell / 1.5 + i / (double)jsize;
  }

  reset_udata<T>(b_ucell, mesh_.totald_size(), location::cell);
  reset_udata<T>(d_ucell, mesh_.totald_size(), location::cell);


  const unsigned int block_size_x = BLOCKSIZEX;
  const unsigned int block_size_y = BLOCKSIZEY;

  const unsigned int nbx = (isize + block_size_x - 1) / block_size_x;
  const unsigned int nby = (jsize + block_size_y - 1) / block_size_y;

  dim3 num_blocks(nbx, nby, 1);
  dim3 block_dim_ilp(block_size_x, block_size_y);
  dim3 block_dim(block_size_x, block_size_y*2);


  std::chrono::high_resolution_clock::time_point t1, t2;

  /*  umesh umesh_(mesh_.compd_size(), mesh_.totald_size(),
               mesh_.nodes_totald_size());
  */
  //  mesh_to_hilbert(umesh_, mesh_);

    const unsigned int mesh_size =
        mesh_.get_elements(location::cell).last_compute_domain_idx();

    const unsigned int nbx1d =
        (mesh_size + (block_size_x * 2) - 1) / (block_size_x * 2);

    dim3 num_blocks1d(nbx1d, 1, 1);
    dim3 block_dim1d(block_size_x * 2, 1);


#ifdef NOPREFETCH
    
#else
  gpuErrchk(cudaMemPrefetchAsync(a_cell, sizeof(T) * total_size_cell,0));
  gpuErrchk(cudaMemPrefetchAsync(b_cell, sizeof(T) * total_size_cell,0));
  gpuErrchk(cudaMemPrefetchAsync(c_cell, sizeof(T) * total_size_cell,0));
  gpuErrchk(cudaMemPrefetchAsync(d_cell, sizeof(T) * total_size_cell,0));
  gpuErrchk(cudaMemPrefetchAsync(fac1_cell, sizeof(T) * total_size_cell,0));
  gpuErrchk(cudaMemPrefetchAsync(fac2_cell, sizeof(T) * total_size_cell,0));
  
  gpuErrchk(cudaMemPrefetchAsync(a_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));
  gpuErrchk(cudaMemPrefetchAsync(b_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));
  gpuErrchk(cudaMemPrefetchAsync(c_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));
  gpuErrchk(cudaMemPrefetchAsync(d_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));
  gpuErrchk(cudaMemPrefetchAsync(fac1_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));
  gpuErrchk(cudaMemPrefetchAsync(fac2_ucell, sizeof(T) * mesh_.totald_size() * num_colors(location::cell) * ksize,0));

  gpuErrchk(cudaMemPrefetchAsync(mesh_.m_cells.m_idx,elements::size_of_array(location::cell,isize,jsize,halo),0));
  gpuErrchk(cudaMemPrefetchAsync(mesh_.m_nodes.m_x,nodes::size_of_mesh_fields<double>(location::vertex,isize,jsize,halo),0));
  gpuErrchk(cudaMemPrefetchAsync(mesh_.m_nodes.m_y,nodes::size_of_mesh_fields<double>(location::vertex,isize,jsize,halo),0));
  
  
  /*  gpuErrchk(cudaMemPrefetchAsync(umesh_.m_cells.m_idx,umesh_.m_cells.m_totald_size * sizeof(size_t),0));
  gpuErrchk(cudaMemPrefetchAsync(umesh_.m_nodes.m_x,umesh_.m_nodes.m_totald_size * sizeof(double),0));
  gpuErrchk(cudaMemPrefetchAsync(umesh_.m_nodes.m_y,umesh_.m_nodes.m_totald_size * sizeof(double),0));
  */
  
  gpuErrchk(cudaDeviceSynchronize());

#endif  

  for (unsigned int t = 0; t < tsteps; t++) {

    //----------------------------------------//
    //----------------  COPY  ----------------//
    //----------------------------------------//

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    copy<<<num_blocks, block_dim_ilp>>>(a_cell, b_cell, init_offset, cstride_cell,
                                    jstride_cell, kstride_cell, ksize, isize,
                                    jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[ucopy_st] += std::chrono::duration<double>(t2 - t1).count();

#ifdef VERIFY
    if (!t) {
      std::cout << "RUNNING COPY" << std::endl;
      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int c = 0; c < num_colors(location::cell); ++c) {
          for (unsigned int j = 0; j < jsize; ++j) {
            for (unsigned int k = 0; k < ksize; ++k) {
              if (b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)] !=
                  a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)]) {
                printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                       b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)],
                       a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)]);
              }
            }
          }
        }
      }
    }
#endif
}
    //----------------------------------------//
    //--------------  COPY  MESH -------------//
    //----------------------------------------//

  for (unsigned int t = 0; t < tsteps; t++) {

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();

    copy_mesh<<<num_blocks1d, block_dim1d>>>(a_cell, b_cell, init_offset,
                                             kstride_cell, ksize, mesh_size);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[ucopymesh_st] += std::chrono::duration<double>(t2 - t1).count();

#ifdef VERIFY 
   if (!t) {
      std::cout << "RUNNING COPY MESH" << std::endl;

      for (unsigned int i = 0; i < isize; ++i) {
        for (unsigned int c = 0; c < num_colors(location::cell); ++c) {
          for (unsigned int j = 0; j < jsize; ++j) {
            for (unsigned int k = 0; k < ksize; ++k) {
              if (b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)] !=
                  a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                 kstride_cell, init_offset)]) {
                printf("Error in (%d,%d,%d) : %f %f\n", (int)i, (int)j, (int)k,
                       b_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)],
                       a_cell[uindex3(i, c, j, k, cstride_cell, jstride_cell,
                                      kstride_cell, init_offset)]);
              }
            }
          }
        }
      }
   }
#endif
}
    //----------------------------------------//
    //-------------  ONCELLS ILP -------------//
    //----------------------------------------//
#ifdef VERIFY
  reset_data<T>(b_cell, location::cell, isize, jsize, ksize, cstride_cell, jstride_cell, kstride_cell, init_offset);
#endif

  for (unsigned int t = 0; t < tsteps; t++) {

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    on_cells_ilp<<<num_blocks, block_dim_ilp>>>(a_cell, b_cell, init_offset,
                                        cstride_cell, jstride_cell,
                                        kstride_cell, ksize, isize, jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[uoncells_ilp_st] += std::chrono::duration<double>(t2 - t1).count();
#ifdef VERIFY
  if (!t) {
      std::cout << "ONCELLS ILP" << std::endl;

      verify_on_cells<T>(b_cell, a_cell, isize, jsize, ksize, cstride_cell,
                         jstride_cell, kstride_cell, init_offset);
    }
#endif
}
    //----------------------------------------//
    //---------------  ONCELLS  --------------//
    //----------------------------------------//
#ifdef VERIFY
    reset_data<T>(b_cell, location::cell, isize, jsize, ksize, cstride_cell, jstride_cell, kstride_cell, init_offset);
#endif
  for (unsigned int t = 0; t < tsteps; t++) {

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    on_cells<<<num_blocks, block_dim>>>(a_cell, b_cell, init_offset,
                                        cstride_cell, jstride_cell,
                                        kstride_cell, ksize, isize, jsize);
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[uoncells_st] += std::chrono::duration<double>(t2 - t1).count();
#ifdef VERIFY 
   if (!t) {
      std::cout << "RUNNING ONCELLS" << std::endl;

      verify_on_cells<T>(b_cell, a_cell, isize, jsize, ksize, cstride_cell,
                         jstride_cell, kstride_cell, init_offset);
    } 
#endif
}
    //----------------------------------------//
    //-------------  ONCELLS MESH ------------//
    //----------------------------------------//
#ifdef VERIFY
    reset_udata<T>(b_ucell, mesh_.totald_size(), location::cell);
#endif
  for (unsigned int t = 0; t < tsteps; t++) {

    gpuErrchk(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();

    on_cells_mesh<<<num_blocks1d, block_dim1d,
                    block_dim1d.x *
                        num_neighbours(location::cell, location::cell) *
                        sizeof(size_t)>>>(
        a_ucell, b_ucell, mesh_.totald_size() * num_colors(location::cell),
        ksize, mesh_size,
        mesh_.get_elements(location::cell).table(location::cell));
    // gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    t2 = std::chrono::high_resolution_clock::now();
    if (t > warmup_step)
      timings[uoncellsmesh_st] +=
          std::chrono::duration<double>(t2 - t1).count();
#ifdef VERIFY
    if (!t) {
      std::cout << "RUNNING ONCELLS MESH" << std::endl;

      verify_on_ucells<T>(b_ucell, a_ucell, ksize, mesh_);
    }
#endif
}
    //----------------------------------------//
    //--------------  HILBERT MESH ------------//
    //----------------------------------------//
  /*
    {
#ifdef VERIFY
      reset_udata<T>(b_ucell, mesh_.totald_size(), location::cell);
#endif

      auto cells = umesh_.get_elements(location::cell);
      auto cell_to_cell = cells.table(location::cell);
      
  for (unsigned int t = 0; t < tsteps; t++) {

      gpuErrchk(cudaDeviceSynchronize());
      t1 = std::chrono::high_resolution_clock::now();
      on_cells_umesh<<<num_blocks1d, block_dim1d,
                       block_dim1d.x *
                           num_neighbours(location::cell, location::cell) *
                           sizeof(size_t)>>>(
          a_ucell, b_ucell, umesh_.totald_size() * num_colors(location::cell),
          ksize, mesh_size,
          umesh_.get_elements(location::cell).table(location::cell));
      gpuErrchk(cudaDeviceSynchronize());

      t2 = std::chrono::high_resolution_clock::now();
      if (t > warmup_step)
        timings[uoncellsmesh_hilbert_st] +=
            std::chrono::duration<double>(t2 - t1).count();
#ifdef VERIFY
      if (!t) {
        std::cout << "RUNNING HILBERT MESH" << std::endl;

        verify_on_ucells<T>(b_ucell, a_ucell, ksize, umesh_);
      }
#endif
    }
}
  */

    //----------------------------------------//
    //-----------  COMPLEX ONCELLS  ----------//
    //----------------------------------------//
    {
  for (unsigned int t = 0; t < tsteps; t++) {
 
     gpuErrchk(cudaDeviceSynchronize());
      t1 = std::chrono::high_resolution_clock::now();
      complex_on_cells<<<num_blocks, block_dim_ilp>>>(
          a_cell, b_cell, c_cell, d_cell, fac1_cell, fac2_cell, init_offset,
          cstride_cell, jstride_cell, kstride_cell, ksize, isize, jsize);
      // gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      t2 = std::chrono::high_resolution_clock::now();
      if (t > warmup_step)
        timings[complex_uoncells_st] +=
            std::chrono::duration<double>(t2 - t1).count();
}
    }
    //----------------------------------------//
    //---------  COMPLEX ONCELLS MESH --------//
    //----------------------------------------//
    {
  for (unsigned int t = 0; t < tsteps; t++) {
 
     gpuErrchk(cudaDeviceSynchronize());
      t1 = std::chrono::high_resolution_clock::now();

      complex_on_cells_mesh<<<num_blocks1d, block_dim1d,
                              block_dim1d.x * num_neighbours(location::cell,
                                                             location::cell) *
                                  sizeof(size_t)>>>(
          a_ucell, b_ucell, c_ucell, d_ucell, fac1_ucell, fac2_ucell,
          mesh_.totald_size() * num_colors(location::cell), ksize, mesh_size,
          mesh_.get_elements(location::cell).table(location::cell));
      // gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      t2 = std::chrono::high_resolution_clock::now();
      if (t > warmup_step)
        timings[complex_uoncellsmesh_st] +=
            std::chrono::duration<double>(t2 - t1).count();
}
    }
    //----------------------------------------//
    //----------  COMPLEX HILBER MESH --------//
    //----------------------------------------//
    /*    {
      auto cells = umesh_.get_elements(location::cell);
      auto cell_to_cell = cells.table(location::cell);
  for (unsigned int t = 0; t < tsteps; t++) {

      gpuErrchk(cudaDeviceSynchronize());
      t1 = std::chrono::high_resolution_clock::now();

      

      complex_on_cells_umesh<<<num_blocks1d, block_dim1d,
                               block_dim1d.x * num_neighbours(location::cell,
                                                              location::cell) *
                                   sizeof(size_t)>>>(
          a_ucell, b_ucell, c_ucell, d_ucell, fac1_ucell, fac2_ucell,
          umesh_.totald_size() * num_colors(location::cell), ksize, mesh_size,
          umesh_.get_elements(location::cell).table(location::cell));
      gpuErrchk(cudaDeviceSynchronize());

      t2 = std::chrono::high_resolution_clock::now();
      if (t > warmup_step)
        timings[complex_uoncellsmesh_hilbert_st] +=
            std::chrono::duration<double>(t2 - t1).count();
	    }
	    }*/
}
