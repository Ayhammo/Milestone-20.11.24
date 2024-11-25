// -*- C++ -*-
/*
==================================================
Authors: A. Mithran; I. Kulakov; M. Zyzak
==================================================
*/

/// use "g++ -O3 -fno-tree-vectorize -msse Matrix.cpp && ./a.out" to run
// Finish SIMDized version. Compare results and time.

#include "fvec/P4_F32vec4.h"    // wrapper of the SSE instruction
#include "utils/TStopWatch.h"


#include <stdlib.h>  // rand
#include <iostream>

const int N = 1000;  // matrix size. Has to be dividable by 4.

const int NIter = 100;  // repeat calculations many times in order to neglect
                        // memory reading time

float a[N][N];       // input array
float c[N][N];       // output array for scalar computations
float c_simd[N][N];  // output array for SIMD computations

template <typename T>  // required calculations
T f(T x) {
  return sqrt(x);
}

void CheckResults(const float a1[N][N], const float a2[N][N]) {
  bool ok = true;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (fabs(a1[i][j] - a2[i][j]) > 1.e-8) ok = false;

  if (ok)
    std::cout << "SIMD and scalar results are the same." << std::endl;
  else
    std::cout << "ERROR! SIMD and scalar results are not the same."
              << std::endl;
}

int main() {
  // fill classes by random numbers
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] =
          float(rand()) / float(RAND_MAX);  // put a random value, from 0 to 1
    }
  }

  /// -- CALCULATE --
  /// SCALAR
  TStopwatch timerScalar;
  for (int ii = 0; ii < NIter; ii++)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        c[i][j] = f(a[i][j]);
      }
    }
  timerScalar.Stop();

  /// SIMD VECTORS
  TStopwatch timerSIMD;
  for (int ii = 0; ii < NIter; ii++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j += 4) {
        F32vec4* vecA = reinterpret_cast<F32vec4*>(&a[i][j]); 
        F32vec4* vecC = reinterpret_cast<F32vec4*>(&c_simd[i][j]);
        *vecC = f(*vecA);
      }
    }
  }
  timerSIMD.Stop();

  double tScal = timerScalar.RealTime() * 1000;
  double tSIMD1 = timerSIMD.RealTime() * 1000;

  std::cout << "Time scalar: " << tScal << " ms " << std::endl;
  std::cout << "Time SIMD:   " << tSIMD1 << " ms, speed up " << tScal / tSIMD1
            << std::endl;

  CheckResults(c, c_simd);

  return 0;
}
