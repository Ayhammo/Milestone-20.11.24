// -*- C++ -*-
/*
==================================================
Authors: A. Mithran; I. Kulakov; M. Zyzak
==================================================
*/
/// use "g++ -O3 -fno-tree-vectorize -msse CheckSum.cpp && ./a.out" to run

// make calculation parallel: a) using SIMD instructions, b) using usual instructions!

#include "fvec/P4_F32vec4.h"    // wrapper of the SSE instruction
#include "utils/TStopWatch.h"

#include <iostream>
using namespace std;

#include <stdlib.h> // rand

const int NIter = 100;

const int N = 4000000; // matrix size. Has to be dividable by 4.
unsigned char str[N];

template <typename T>
T Sum(const T* data, const int N)
{
    T sum = 0;

    for (int i = 0; i < N; ++i)
        sum = sum ^ data[i];
    return sum;
}

int main()
{
    // fill string by random values
    for (int i = 0; i < N; i++) {
        str[i] = 256 * (double(rand()) /
                        RAND_MAX); // put a random value, from 0 to 255
    }

    /// -- CALCULATE --

    /// SCALAR

    unsigned char sumS = 0;
    TStopwatch timerScalar;
    for (int ii = 0; ii < NIter; ii++)
        sumS = Sum<unsigned char>(str, N);
    timerScalar.Stop();

    /// SIMD

    unsigned char sumV = 0;

    TStopwatch timerSIMD;
    timerSIMD.Start();
    for (int i = 0: i < NIter; i++){
        F32vec4 ssum(0.0f); //F43vec4 Initialize for 4 Float
        float* sdata = reinterpret_cast<float*>(str); //str to float
        int vcount = N/sizeof(F32vec4);//Count the Float Block
        for (int i = 0; i < vcount; i++){ //Through the Blocks and save XOR to SIMD ssum
            F32vec4 dataBlock(sdata[i]);
            ssum = ssum ^ dataBlock;
        }
        float reduced[4];//4 Floats in SIMD to array reduced
        ssum.store(reduced);
        for (int j = 0; j < 4; j++){ //XOR to end result sumV
            sumV ^= static_cast<unsigned char>(reduced[i]);
        }
        
    }
    timerSIMD.Stop();

    /// SCALAR INTEGER

    unsigned char sumI = 0;

    TStopwatch timerINT;
    timerINT.Start();
    for (int i = 0; i < NIter; i++){
        unsigned int* idata = reinterpret_cast<unsigned int*>(str); //Transfer str to unsigned int array
        int bcount = N/sizeof(unsigned int); //Count the Number of Blocks
        unsigned int summ = 0;
        for (int j = 0; j < bcount; j++){ //XOR Ops on Blocks
            summ ^= data[i];
        }
        sumI = static_cast<unsigned char>(summ^(summ >> 8)^(summ >> 16)^(summ >> 24));//jede in 32bits XOR and get a one Byte Result
    }
    timerINT.Stop();

    /// -- OUTPUT --

    double tScal = timerScalar.RealTime() * 1000;
    double tINT = timerINT.RealTime() * 1000;
    double tSIMD = timerSIMD.RealTime() * 1000;

    cout << "Time scalar: " << tScal << " ms " << endl;
    cout << "Time INT:   " << tINT << " ms, speed up " << tScal / tINT << endl;
    cout << "Time SIMD:   " << tSIMD << " ms, speed up " << tScal / tSIMD
         << endl;

    // cout << static_cast<int>(sumS) << " " << static_cast<int>(sumV) << endl;
    if (sumV == sumS && sumI == sumS)
        std::cout << "Results are the same." << std::endl;
    else
        std::cout << "ERROR! Results are not the same." << std::endl;

    return 0;
}