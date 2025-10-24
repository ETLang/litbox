#include "UnityCG.cginc"

struct appdata
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
};

struct v2f
{
    float2 uv : TEXCOORD0;
    float2 farm_uv : TEXCOORD1;
    float4 vertex : SV_POSITION;
    float3 normal : NORMAL;
};

sampler2D _MainTex;
float4x4 _FarmlandTransform;
float _FarmlandRowCount;
float _LeftHeight;
float _PeakHeight;
float _RightHeight;
float _Pinch;
float4 _BaseColor;
float _SpecularPower;
float4 _LeftAmbience;
float4 _RightAmbience;
float4 _Haze;
float4 _SpecularColor;
float4 _SpecularSource;
float _ViewXShift;
float4 _FuzzColor;
float _FuzzLength;
float _ZOffset;
float _substrateDensity;

v2f hill_vert(appdata v)
{
    const float perspective = 2;
    const float pi = 3.141592564f;
    const float xLeft = -5;
    const float xRight = 5;

    const float gaussianLimit = 2;
    const float gaussianLowerbound = exp(-pow(gaussianLimit, 2));

    float x = lerp(xLeft, xRight, v.uv.x) * lerp(perspective, 1, 1 - pow(1 - v.uv.y, 2));;
    float wx = lerp(-gaussianLimit, gaussianLimit, v.uv.x);
    float w = exp(-pow(wx, 2));
    float dy = -2 * wx * w;
    w = (w - gaussianLowerbound) / (1 - gaussianLowerbound);
    float top;

    if (v.uv.x < 0.5)
    {
        top = lerp(_LeftHeight, _PeakHeight, w);
        dy *= (_PeakHeight - _LeftHeight);
    }
    else
    {
        top = lerp(_RightHeight, _PeakHeight, w);
        dy *= (_PeakHeight - _RightHeight);
    }
    float y = top * (exp(-pow(1 - v.uv.y, 2)) - exp(-1)) / (1 - exp(-1)); // v.uv.y;

                // Compute normal from dy
    float3 n;
    if (abs(dy) < 1e-6)
        n = float3(0, 1, 0);
    else
        n = normalize(float3((2 * gaussianLimit) / (xRight - xLeft), -1 / dy, 0)) * sign(-dy);
    n = lerp(normalize(float3(0, 1, -2)), n, pow(v.uv.y, 2));

                // Send to frag
    v2f o;
    o.vertex = UnityObjectToClipPos(float4(x, y, 0, 1));
    o.vertex /= o.vertex.w;
    o.vertex.z += _ZOffset;
    o.uv.xy = v.uv;
    o.farm_uv = mul(_FarmlandTransform, float4(v.uv, 0, 1)).xy;
    o.normal = n;
    return o;
}

float3 tone_map(float3 x)
{
    return smoothstep(-4, 2, log10(x));
}

struct gbuffer_output
{
    float4 albedo : SV_Target0;
    float4 transmissibility : SV_Target1;
    float4 normal : SV_Target2;
};

gbuffer_output hill_frag(v2f i)
{
    float2 farmuv = i.farm_uv;
    if (_FarmlandRowCount != -1)
    {
        farmuv.x = max(0, min(farmuv.x, _FarmlandRowCount));
    }
    float4 farmland_color = tex2D(_MainTex, farmuv.yx);
    clip(farmland_color.a - 1e-3);

    float3 normal = i.normal;
    normal.z = min(normal.z, 0);

    float3 diffuse_light = lerp(_LeftAmbience.rgb, _RightAmbience.rgb, (i.normal.x + 1) / 2);

    float fuzz_factor = 1 - pow(abs(i.normal.z), _FuzzLength);
    float3 color = lerp(farmland_color.rgb, _FuzzColor.rgb, fuzz_factor);
    float3 diffuse_color = diffuse_light * color;

    const float3 view_vec = normalize(float3((i.uv.x * 2 - 1) * _ViewXShift, 0, 1));
    const float3 specular_source = normalize(_SpecularSource.xyz);
    float specular_factor = pow(saturate(dot(normalize(reflect(view_vec, i.normal)), specular_source)), _SpecularPower);

    float3 specular_color = _SpecularColor.rgb * specular_factor;
    
    float3 final_color = lerp(diffuse_color + specular_color, _Haze.rgb, _Haze.a);

    float t = 1 - _substrateDensity * farmland_color.a / sqrt(_ScreenParams.x * _ScreenParams.y) * 100;
    gbuffer_output output;
    output.albedo = float4(final_color, 1) * farmland_color.a;
    output.transmissibility = float4(t, t, 0, 1);
    output.normal = float4(0, 1, 0, 0);
    return output;
}
