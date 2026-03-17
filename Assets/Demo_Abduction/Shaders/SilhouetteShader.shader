Shader "Abductor/SilhouetteShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Color ("Color", Color) = (1, 1, 1, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            ZWrite Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float3 normal : NORMAL;
                float4 vertex : SV_POSITION;
                float2 screen_uv : TEXCOORD2;
            };

            float4 _Color;
            float _substrateDensity;
            float _particleAlignment;
            float _heightScale;
            float _Metallic;
            float3 _Ambience;

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _NormalTex;
            sampler2D _lightMap;
            float4x4 _LightingUVTransform;
            float _ZOffset;
            bool _isRayTracing;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.vertex /= o.vertex.w;
                o.vertex.z += _ZOffset;
                o.normal = UnityObjectToWorldNormal(v.normal) * _heightScale;
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                float2 ss = o.vertex.xy;
                // ss.y *= -1;
                // ss++;
                // ss /= 2;
                o.screen_uv = mul(_LightingUVTransform, float4(ss, 0, 1)).xy;
                //o.screen_uv = mul(float4(ss, 0, 1), _LightingUVTransform).xy;
                return o;
            }

            float4 frag (v2f i) : SV_Target
            {
                // sample the texture
                //return float4(frac(i.screen_uv), 0, 1);
                float3 light = _Ambience + tex2Dlod(_lightMap, float4(frac(i.screen_uv), 0, 0)).rgb;
                //return float4(light, 1);
                float4 col = tex2D(_MainTex, i.uv);
                col *= _Color;
                float3 finalColor = col.rgb * light + light * _Metallic;
                return float4(finalColor, col.a);
            }
            ENDCG
        }
    }
}
