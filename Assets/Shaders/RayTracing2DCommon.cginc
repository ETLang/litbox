#include "UnityCG.cginc"
//#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

#define PI 3.141592654

// Foundation

struct appdata_common
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
};

struct v2f_common
{
    float2 uv : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

v2f_common vert_common(appdata_common v)
{
    v2f_common o;
    o.vertex = v.vertex;
    o.vertex.y *= sign(UNITY_MATRIX_VP[1][1]);   // HACK OMG gross
    o.uv = v.uv;

    return o;
}

Texture2D<float4> _MainTex;
sampler sampler_MainTex;
half4 _MainTex_ST;
float _MainTexLOD;
float4 _ColorMod;

#define SAMPLE(Map, UV) SAMPLE_TEXTURE2D(Map, sampler##Map, (UV))
#define SAMPLE_LOD(Map, UV, LOD) Map.SampleLevel(sampler##Map, UV, LOD)
#define SAMPLE_MAIN(UV) SAMPLE_LOD(_MainTex, (UV), _MainTexLOD)

float4 frag_blit(v2f_common i) : SV_Target
{
    return SAMPLE_MAIN(i.uv);
}

float4 frag_blit_and_modulate(v2f_common i) : SV_Target
{
    //return float4(1,0,0,1);
    return SAMPLE_MAIN(i.uv) *_ColorMod;
}

float2 TextureSize(Texture2D tex, uint mipLevel = 0)
{
    float w, h, _;
    tex.GetDimensions(mipLevel, w, h, _);
    return float2(w, h);
}

float Intensity(float3 color)
{
    return dot(float3(0.299, 0.587, 0.114), color);
}

float3 ToGrayscale(float3 color)
{
    return Intensity(color).xxx;
}

// Utilities

// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application

struct RandomContext
{
    uint4 state;
    float value;
};

RandomContext CreateRandomContext(uint4 seed)
{
    RandomContext ret;
    ret.state = seed;
    ret.value = 2.3283064365387e-10 * (ret.state.x ^ ret.state.y ^ ret.state.z ^ ret.state.w);
    return ret;
}

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C)
{
    return (A * z + C);
}

RandomContext NextRandom(RandomContext ctx)
{
    RandomContext next;

    next.state.x = TausStep(ctx.state.x, 13, 19, 12, 4294967294);
    next.state.y = TausStep(ctx.state.y, 2, 25, 4, 4294967288);
    next.state.z = TausStep(ctx.state.z, 3, 11, 17, 4294967280);
    next.state.w = LCGStep(ctx.state.w, 1664525, 1013904223);
    next.value = 2.3283064365387e-10 * (next.state.x ^ next.state.y ^ next.state.z ^ next.state.w);

    return next;
}

//#define INIT_RANDOM(seed) RandomContext __rc = CreateRandomContext(seed);
//#define RAND_NEXT() __rc = NextRandom(__rc)

#define DECLARE_LUT(type, name) \
Texture2D<type> name;           \
SamplerState sampler##name;     \
float2 lut_window_##name;

#define SampleLUT(name,u) name.SampleLevel(sampler##name, float2(dot(float2(1,u),lut_window_##name),0), 0)
