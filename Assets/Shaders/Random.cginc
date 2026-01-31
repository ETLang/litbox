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

#endif // _RANDOM_