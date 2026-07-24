#ifndef _RANDOM_
#define _RANDOM_

// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

struct Random
{
    uint4 state;
    float value;

    void Init(uint4 seed) {
        state = seed;
        value = 2.3283064365387e-10 * (state.x ^ state.y ^ state.z ^ state.w);
    }

    float Next()
    {
        state.x = _TausStep(state.x, 13, 19, 12, 4294967294);
        state.y = _TausStep(state.y, 2, 25, 4, 4294967288);
        state.z = _TausStep(state.z, 3, 11, 17, 4294967280);
        state.w = _LCGStep(state.w, 1664525, 1013904223);
        value = 2.3283064365387e-10 * (state.x ^ state.y ^ state.z ^ state.w);
    
        return value;
    }

    float Next(float lo, float hi) { return lo + Next() * (hi - lo); }
    float2 Next2() { return float2(Next(), Next()); }
    float3 Next3() { return float3(Next(), Next(), Next()); }
    float4 Next4() { return float4(Next(), Next(), Next(), Next()); }
    
    float2 NextDirection() {
        float theta = Next() * 2 * 3.141592654f;
        float2 dir;
        sincos(theta, dir.x, dir.y);
        return dir;
    }

    float2 NextCircle() {
        return NextDirection() * sqrt(Next());
    }
    
    
    uint _TausStep(uint z, int S1, int S2, int S3, uint M)
    {
        uint b = (((z << S1) ^ z) >> S2);
        return (((z & M) << S3) ^ b);
    }

    uint _LCGStep(uint z, uint A, uint C)
    {
        return (A * z + C);
    }
};

Random CreateRandom(uint4 seed)
{
    Random ret;
    ret.Init(seed);
    return ret;
}

// ---------------------------------------------------------------------------------------------
// Ported from David Hoskin's integer-hash family. Created by David Hoskins, May 2018.
// https://www.shadertoy.com/view/XdGfRR
// Licensed under Creative Commons Attribution-ShareAlike 4.0 International
// (https://creativecommons.org/licenses/by-sa/4.0/).
//
// Naming: hash(out)(in), e.g. hash23 takes 2 inputs and produces a 3-component output. Unlike the
// WGSL port of this same family (Random.wgsl), HLSL supports overloading, so the uint/uintN and
// float/floatN overloads keep their original shared names, resolved by argument type exactly as
// in the GLSL source. Otherwise this is a direct, unmodified port, including hash12's odd
// `/ 0xffffffffu` divisor (every other function normalizes with the `2.328306437080797e-10`
// constant instead) and hash44(float4)'s float3 return that silently drops the w channel - both
// quirks are present in the original source, kept here for fidelity.
//---------------------------------------------------------------------------------------------------------------
float hash11(uint q)
{
    uint2 n = q * uint2(1597334673u, 3812015801u);
    q = (n.x ^ n.y) * 1597334673u;
    return (float)q * 2.328306437080797e-10;
}

float hash11(float p)
{
    uint2 n = (uint)(int)p * uint2(1597334673u, 3812015801u);
    uint q = (n.x ^ n.y) * 1597334673u;
    return (float)q * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float hash12(uint2 q)
{
    q *= uint2(1597334673u, 3812015801u);
    uint n = (q.x ^ q.y) * 1597334673u;
    return (float)n / (float)0xffffffffu;
}

float hash12(float2 p)
{
    uint2 q = (uint2)(int2)p * uint2(1597334673u, 3812015801u);
    uint n = (q.x ^ q.y) * 1597334673u;
    return (float)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float hash13(uint3 q)
{
    q *= uint3(1597334673u, 3812015801u, 2798796415u);
    uint n = (q.x ^ q.y ^ q.z) * 1597334673u;
    return (float)n * 2.328306437080797e-10;
}

float hash13(float3 p)
{
    uint3 q = (uint3)(int3)p * uint3(1597334673u, 3812015801u, 2798796415u);
    uint n = (q.x ^ q.y ^ q.z) * 1597334673u;
    return (float)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float hash14(uint4 q)
{
    q *= uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    uint n = (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
    return (float)n * 2.328306437080797e-10;
}

float hash14(float4 p)
{
    uint4 q = (uint4)(int4)p * uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    uint n = (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
    return (float)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float2 hash21(uint q)
{
    uint2 n = q * uint2(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * uint2(1597334673u, 3812015801u);
    return (float2)n * 2.328306437080797e-10;
}

float2 hash21(float p)
{
    uint2 n = (uint)(int)p * uint2(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * uint2(1597334673u, 3812015801u);
    return (float2)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float2 hash22(uint2 q)
{
    q *= uint2(1597334673u, 3812015801u);
    q = (q.x ^ q.y) * uint2(1597334673u, 3812015801u);
    return (float2)q * 2.328306437080797e-10;
}

float2 hash22(float2 p)
{
    uint2 q = (uint2)(int2)p * uint2(1597334673u, 3812015801u);
    q = (q.x ^ q.y) * uint2(1597334673u, 3812015801u);
    return (float2)q * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float2 hash23(uint3 q)
{
    q *= uint3(1597334673u, 3812015801u, 2798796415u);
    uint2 n = (q.x ^ q.y ^ q.z) * uint2(1597334673u, 3812015801u);
    return (float2)n * 2.328306437080797e-10;
}

float2 hash23(float3 p)
{
    uint3 q = (uint3)(int3)p * uint3(1597334673u, 3812015801u, 2798796415u);
    uint2 n = (q.x ^ q.y ^ q.z) * uint2(1597334673u, 3812015801u);
    return (float2)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float3 hash31(uint q)
{
    uint3 n = q * uint3(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)n * 2.328306437080797e-10;
}

float3 hash31(float p)
{
    uint3 n = (uint)(int)p * uint3(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float3 hash32(uint2 q)
{
    uint3 n = q.xyx * uint3(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)n * 2.328306437080797e-10;
}

float3 hash32(float2 q)
{
    uint3 n = (uint3)(int3)q.xyx * uint3(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)n * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float3 hash33(uint3 q)
{
    q *= uint3(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)q * 2.328306437080797e-10;
}

float3 hash33(float3 p)
{
    uint3 q = (uint3)(int3)p * uint3(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * uint3(1597334673u, 3812015801u, 2798796415u);
    return (float3)q * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
float4 hash44(uint4 q)
{
    q *= uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    q = (q.x ^ q.y ^ q.z ^ q.w) * uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    return (float4)q * 2.328306437080797e-10;
}

// NOTE: returns float3, not float4 - matches the original source exactly, which drops the w
// channel here (the uint4 overload above returns the full float4).
float3 hash44(float4 p)
{
    uint4 q = (uint4)(int4)p * uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    q = (q.x ^ q.y ^ q.z ^ q.w) * uint4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    return (float3)q.xyz * 2.328306437080797e-10;
}

#endif // _RANDOM_