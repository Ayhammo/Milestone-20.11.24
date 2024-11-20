// -*- C++ -*-
/*
==================================================
Authors: A. Mithran; I. Kulakov; M. Zyzak
==================================================
*/
/// use "g++ -O3 -fno-tree-vectorize -msse QuadraticEqn.cpp && ./a.out" to run
/// Note:
/// __m128 - SIMD vector
/// SIMD intrinsics:
/// _mm_set_ps(f3,f2,f1,f0) - write 4 floats into the SIMD vector
/// note, that the order of entries is inversed
/// _mm_set_ps1(a) - write float "a" into the SIMD vector
/// _mm_add_ps(a,b) - a+b
/// _mm_sub_ps(a,b) - a-b
/// _mm_mul_ps(a,b) - a*b
/// _mm_div_ps(a,b) - a/b
/// _mm_sqrt_ps(a) - sqrt(a)
/// type2* pointer2 = reinterpret_cast<type2*>( pointer1 ) - change pointer type

#include "fvec/P4_F32vec4.h"    // wrapper of the SSE instruction
#include "utils/TStopWatch.h"

#include <cmath>
#include <iostream>

#include <stdlib.h> // rand

static const int NVectors = 1000000;
static const int N = NVectors * fvecLen;

static const int NIterOut = 1000;

void CheckResults(const float* yScalar, const float* ySIMD, const int NSIMD)
{
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabs(yScalar[i] - ySIMD[i]) > yScalar[i] * 0.001) {
            ok = false;
            std::cout << i<<" " << yScalar[i] << " " << ySIMD[i] << " " << fabs(yScalar[i] - ySIMD[i])<<std::endl;
        }
    if (!ok)
        std::cout << "ERROR! SIMD" << NSIMD
                  << " and scalar results are not the same." << std::endl;
    else
      std::cout << "SIMD" << NSIMD << " and scalar results are the same." << std::endl;
}

int main()
{
    //input data
    float* a = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* b = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* c = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    //output data
    float* x = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* x_simd1 = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* x_simd2 = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* x_simd3 = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));
    float* x_simd4 = (float*) _mm_malloc(sizeof(float)*N,16);// __attribute__ ((aligned(16)));

    // fill parameters by random numbers
    for (int i = 0; i < N; i++) {
        a[i] =
            0.01 + float(rand()) / float(RAND_MAX); // put a random value, from 0.01 to 1.01 (a has not to be equal 0)
        b[i] = float(rand()) / float(RAND_MAX);
        c[i] = -float(rand()) / float(RAND_MAX);
    }

    /// -- CALCULATE --

    // scalar calculations
    TStopwatch timerScalar;
    for (int io = 0; io < NIterOut; io++)
        for (int i = 0; i < N; i++) {
            float det = b[i] * b[i] - 4 * a[i] * c[i];
            x[i] = (-b[i] + sqrt(det)) / (2 * a[i]);
        }
    timerScalar.Stop();

    // SIMD clculations with SIMD intrinsics and data copy
    TStopwatch timerSIMD;
    for (int io = 0; io < NIterOut; io++)
        for (int i = 0; i < NVectors; i++) {
            ///__put your code here__
            /// copy coefficients b and c
            __m128 bVec = _mm_load_ps(&b[i*fvecLen]);
            __m128 cVec = _mm_load_ps(&c[i*fvecLen]);
            __m128 aVec = _mm_load_ps(&a[i*fvecLen]);
            

            ///__put your code here__
            /// put the code, which calculates the root
            __m128 bbVec = _mm_mul_ps(bVec, bVec);
            __m128 ac4Vec = _mm_mul_ps(_mm_set_ps1(4.0f), _mm_mul_ps(aVec, cVec));
            __m128 detVec = _mm_sub_ps(bbVec, ac4Vec);

            __m128 sqrtVec = _mm_sqrt_ps(detVec);
            __m128 sqrtSubBVec = _mm_sub_ps(sqrtVec, bVec);
            __m128 a2Vec = _mm_mul_ps(_mm_set_ps1(2.0f), aVec);
            __m128 xV = _mm_div_ps(sqrtSubBVec, a2Vec);



            // copy output data
            for(int iE=0; iE<fvecLen; iE++)
              x_simd1[i*fvecLen+iE] = (reinterpret_cast<float*>(&xV))[iE];

        }
    timerSIMD.Stop();

    // SIMD clculations with SIMD intrinsics and reinterpret_cast
    TStopwatch timerSIMD2;
    for (int io = 0; io < NIterOut; io++)
        for (int i = 0; i < N; i += fvecLen) {
            ///__put your code here__
            /// cast coefficients b and c
            __m128* bVec = reinterpret_cast<__m128*>(&b[i]);
            __m128* cVec = reinterpret_cast<__m128*>(&c[i]);
            __m128* aVec = reinterpret_cast<__m128*>(&a[i]); 
            __m128* xV = reinterpret_cast<__m128*>(&x_simd2[i]); 


            ///__put your code here__
            /// put the code, which calculates the root
            __m128 bbVec = _mm_mul_ps(*bVec, *bVec);
            __m128 ac4Vec = _mm_mul_ps(_mm_set_ps1(4), _mm_mul_ps(*aVec, *cVec));
            __m128 detVec = _mm_sub_ps(bbVec, ac4Vec);

            __m128 sqrtSubBVec = _mm_sub_ps(_mm_sqrt_ps(detVec), *bVec);
            *xV = _mm_div_ps(sqrtSubBVec, _mm_mul_ps(_mm_set_ps1(2.0f), *aVec));
        }
    timerSIMD2.Stop();

    // SIMD clculations with headers and data copy
    TStopwatch timerSIMD3;
    for (int io = 0; io < NIterOut; io++) {
        for (int i = 0; i < NVectors; i++) {
            // copy input data
            ///__put your code here__
            /// copy coefficients b and c
            F32vec4 bVec = F32vec4(b[i*fvecLen], b[i*fvecLen+1], b[i*fvecLen+2], b[i*fvecLen+3]);
            F32vec4 cVec = F32vec4(c[i*fvecLen], c[i*fvecLen+1], c[i*fvecLen+2], c[i*fvecLen+3]);
            F32vec4 aVec = F32vec4(a[i*fvecLen], a[i*fvecLen+1], a[i*fvecLen+2], a[i*fvecLen+3]);

            ///__put your code here__
            /// put the code, which calculates the root
            F32vec4 detVec = (bVec * bVec) - (4.0f * aVec * cVec);

            F32vec4 xV = (sqrt(detVec) - bVec) / (2.0f * aVec);

            // copy output data
                  for(int iE=0; iE<fvecLen; iE++)
                    x_simd3[i*fvecLen+iE] = xV[iE];
        }
    }
    timerSIMD3.Stop();

    // SIMD clculations with headers and reinterpret_cast
    TStopwatch timerSIMD4;
    for (int io = 0; io < NIterOut; io++)
        for (int i = 0; i < N; i += fvecLen) {
            ///__put your code here__
            /// cast coefficients b and c
            F32vec4* bVec = reinterpret_cast<F32vec4*>(&b[i]);
            F32vec4* cVec = reinterpret_cast<F32vec4*>(&c[i]);
            F32vec4* aVec = reinterpret_cast<F32vec4*>(&a[i]);
            F32vec4* xV = reinterpret_cast<F32vec4*>(&x_simd4[i]);

            ///__put your code here__
            /// put the code, which calculates the root
            F32vec4 detVec = (*bVec * *bVec) - (F32vec4(4.0f) * *aVec * *cVec);

            *xV = (sqrt(detVec) - *bVec) / (F32vec4(2.0f) * *aVec);
        }
    timerSIMD4.Stop();

    double tScal = timerScalar.RealTime() * 1000;
    double tSIMD1 = timerSIMD.RealTime() * 1000;
    double tSIMD2 = timerSIMD2.RealTime() * 1000;
    double tSIMD3 = timerSIMD3.RealTime() * 1000;
    double tSIMD4 = timerSIMD4.RealTime() * 1000;
    std::cout << "Time scalar: " << tScal << " ms " << std::endl;
    std::cout << "Time SIMD1:   " << tSIMD1 << " ms, speed up "
              << tScal / tSIMD1 << std::endl;
    std::cout << "Time SIMD2:   " << tSIMD2 << " ms, speed up "
              << tScal / tSIMD2 << std::endl;
    std::cout << "Time SIMD3:   " << tSIMD3 << " ms, speed up "
              << tScal / tSIMD3 << std::endl;
    std::cout << "Time SIMD4:   " << tSIMD4 << " ms, speed up "
              << tScal / tSIMD4 << std::endl;

    //compare SIMD and scalar results
    CheckResults(x, x_simd1, 1);
    CheckResults(x, x_simd2, 2);
    CheckResults(x, x_simd3, 3);
    CheckResults(x, x_simd4, 4);

    return 1;
}
