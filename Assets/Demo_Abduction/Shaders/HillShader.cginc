#include "UnityCG.cginc"
#include "../../Shaders/ToneMapping.cginc"

struct appdata
{
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
};

struct v2f
{
    float2 uv : TEXCOORD0;
    float2 farm_uv : TEXCOORD1;
    float2 screen_uv : TEXCOORD2;
    float4 vertex : SV_POSITION;
    float3 normal : NORMAL;
    float3 sim_normal : TEXCOORD3;
};

sampler2D _MainTex;
sampler2D _diffuseLightMap;
float4x4 _FarmlandTransform;
float4x4 _LightingUVTransform;
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
float _RayTracingVerticalOffset;
float _substrateDensity;
bool _isRayTracing;

float2 hill_shape(float2 raw, out float dy)
{
    const float perspective = 2;
    const float pi = 3.141592564f;
    const float xLeft = -5;
    const float xRight = 5;

    const float gaussianLimit = 2;
    const float gaussianLowerbound = exp(-pow(gaussianLimit, 2));

    float x = lerp(xLeft, xRight, raw.x) * lerp(perspective, 1, 1 - pow(1 - raw.y, 2));;
    float wx = lerp(-gaussianLimit, gaussianLimit, raw.x);
    float w = exp(-pow(wx, 2));
    dy = -2 * wx * w;
    w = (w - gaussianLowerbound) / (1 - gaussianLowerbound);
    float top;

    if (raw.x < 0.5)
    {
        top = lerp(_LeftHeight, _PeakHeight, w);
        dy *= (_PeakHeight - _LeftHeight);
    }
    else
    {
        top = lerp(_RightHeight, _PeakHeight, w);
        dy *= (_PeakHeight - _RightHeight);
    }
    float y = top * (exp(-pow(1 - raw.y, 2)) - exp(-1)) / (1 - exp(-1));

    return float2(x, y);
}

v2f hill_vert(appdata v)
{
    float dy;
    float2 pos = hill_shape(v.uv, dy);
    
    const float xLeft = -5;
    const float xRight = 5;

    const float gaussianLimit = 2;

    // Compute cartoon normal from dy
    float3 cn;
    if (abs(dy) < 1e-6)
        cn = float3(0, 1, 0);
    else
        cn = normalize(float3((2 * gaussianLimit) / (xRight - xLeft), -1 / dy , 0)) * sign(-dy);
    cn = lerp(normalize(float3(0, 1, -2)), cn, pow(v.uv.y, 2));

    // Compute simulation normal from smooth derivative
    float ignore;
    float2 tangent = hill_shape(v.uv + float2(1e-3, 0), ignore) - pos;
    float2 n = float2(-tangent.y, tangent.x);

    // Send to frag
    v2f o;
    o.vertex = UnityObjectToClipPos(float4(pos, 0, 1));
    o.vertex /= o.vertex.w;
    o.vertex.z += _ZOffset;
    o.screen_uv = mul(_LightingUVTransform, float4(o.vertex.xy, 0, 1)).xy;
    //o.screen_uv.y = 1 - o.screen_uv.y;
    o.screen_uv.y -= 0.01;
    o.uv.xy = v.uv;
    o.farm_uv = mul(_FarmlandTransform, float4(v.uv, 0, 1)).xy;
    o.normal = cn;
    o.sim_normal = UnityObjectToWorldNormal(normalize(float3(n, 0)));

    if(_isRayTracing) {
        o.vertex.y += _RayTracingVerticalOffset; // Push down slightly to avoid annoying boundary artifacts
    }

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
    ToneMappingShape tone_shape = {
        -0.8,
        {1, 1, 1},
        {-3, -3, -6}
    };

    float2 farmuv = i.farm_uv;
    if (_FarmlandRowCount != -1)
    {
        farmuv.x = max(0, min(farmuv.x, _FarmlandRowCount));
    }
    float4 farmland_color = tex2D(_MainTex, farmuv.yx);
    clip(farmland_color.a - 1e-3);

    float3 normal = i.normal;
    normal.z = min(normal.z, 0);

    float3 diffuse_light_map = tex2D(_diffuseLightMap, frac(i.screen_uv)).rgb;
    float3 diffuse_light = diffuse_light_map + lerp(_LeftAmbience.rgb, _RightAmbience.rgb, (i.normal.x + 1) / 2);

    float fuzz_factor = 1 - pow(abs(i.normal.z), _FuzzLength);
    float3 color = lerp(farmland_color.rgb, _FuzzColor.rgb, fuzz_factor);
    float3 diffuse_color = ToneMap_UE5(diffuse_light, tone_shape) * color;

    const float3 view_vec = normalize(float3((i.uv.x * 2 - 1) * _ViewXShift, 0, 1));
    const float3 specular_source = normalize(_SpecularSource.xyz);
    float specular_factor = pow(saturate(dot(normalize(reflect(view_vec, i.normal)), specular_source)), _SpecularPower);

    float3 specular_color = _SpecularColor.rgb * specular_factor;
    
    float3 final_color = lerp(diffuse_color + specular_color, _Haze.rgb, _Haze.a);

    float t = 1 - _substrateDensity * farmland_color.a / sqrt(_ScreenParams.x * _ScreenParams.y) * 100;
    gbuffer_output output;

    output.albedo = float4(final_color, 1) * farmland_color.a;

    output.transmissibility = float4(t, t, 0, 1);
    output.normal = float4(i.sim_normal, 0.1);
    return output;
}
