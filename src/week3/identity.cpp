#include<iostream>

  /**
   * @brief Identity operation with m=16 and n=16.
   * @param a       Pointer to column-major matrix A.
   * @param b       Pointer to matrix B.
   * @param ld_a    Leading dimension of A.
   * @param ld_b    Leading dimension of B.
   * @param trans_b Column-major B if 0, row-major B if 1. 
   **/
  void identity_16_16( float const * a,
                       float       * b,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int32_t       trans_b );