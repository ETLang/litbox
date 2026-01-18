#ifndef _PHOTONER_COMMON_
#define _PHOTONER_COMMON_

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

#define TEXTURE_SIZE_TEMPLATE(type) \
float2 TextureSize(type tex) { \
    float w, h; \
    tex.GetDimensions(w, h); \
    return float2(w, h); }

TEXTURE_SIZE_TEMPLATE(Texture2D<uint>)
TEXTURE_SIZE_TEMPLATE(Texture2D<uint2>)
TEXTURE_SIZE_TEMPLATE(Texture2D<uint3>)
TEXTURE_SIZE_TEMPLATE(Texture2D<uint4>)
TEXTURE_SIZE_TEMPLATE(Texture2D<float>)
TEXTURE_SIZE_TEMPLATE(Texture2D<float2>)
TEXTURE_SIZE_TEMPLATE(Texture2D<float3>)
TEXTURE_SIZE_TEMPLATE(Texture2D<float4>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<uint>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<uint2>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<uint3>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<uint4>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<float>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<float2>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<float3>)
TEXTURE_SIZE_TEMPLATE(RWTexture2D<float4>)

float2 TextureSize(Texture2D tex, uint mipLevel)
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

float cross2D(float2 a, float2 b)
{
    return dot(a, float2(-b.y, b.x));
}

float ExponentialInterpolation(float u, float d_left, float d_right) {
    return (1-u) - pow(1-u, d_left+1) + pow(u, d_right+1);
}

// Utilities

#define DECLARE_LUT(type, name) \
Texture2D<type> name;           \
SamplerState sampler##name;     \
float2 lut_window_##name;

#define DECLARE_LUT_2D(type, name)  \
Texture2D<type> name;               \
SamplerState sampler##name;         \
float4 lut_window_##name;

#define DECLARE_LUT_3D(type, name)  \
Texture3D<type> name;               \
SamplerState sampler##name;         \
float4 lut_window_##name;           \
float2 lut_slice_window_##name;

#define SampleLUT(name,u) name.SampleLevel(sampler##name, float2(dot(float2(1,u),lut_window_##name),0), 0)

#define SampleLUT2D(name,uv) name.SampleLevel(sampler##name, float2( \
    dot(float2(1,uv.x),lut_window_##name.xy), \
    dot(float2(1,uv.y),lut_window_##name.zw)), 0)

#define SampleLUT3D(name,uvw) name.SampleLevel(sampler##name, float3( \
    dot(float2(1,uvw.x),lut_window_##name.xy), \
    dot(float2(1,uvw.y),lut_window_##name.zw), \
    dot(float2(1,uvw.z),lut_slice_window_##name)), 0)

#define NUMTHREADS_2D 8,8,1
#define NUMTHREADS_1D 64,1,1

SamplerState sampler_point_clamp;
SamplerState sampler_linear_clamp;

#endif // _PHOTONER_COMMON_