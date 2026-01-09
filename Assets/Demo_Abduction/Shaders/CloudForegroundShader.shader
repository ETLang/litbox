Shader "Abduction/CloudForeground"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Color ("Color", Color) = (1,1,1,1) 
        _ForegroundAmbientTex ("Foreground Ambient Texture", 2D) = "white" {}
        _ForegroundAmbientColor ("Foreground Ambient Color", Color) = (0,0,0,1)
        _ObscurityStrength ("Obscurity Strength", Float) = 1.0
        _ObscurityVariation ("Obscurity Variation", Float) = 3.0
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            Blend One OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "../../Shaders/ToneMapping.cginc"

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
                float2 screen_uv : TEXCOORD1;
            };

            float4 _MainTex_ST;
            sampler2D _MainTex;
            sampler2D _ForegroundAmbientTex;
            sampler2D _ForegroundSimulationTex;
            sampler2D _TransmissibilityTex;
            int _ForegroundSimulationLOD;
            float4x4 _ForegroundSimulationUVTransform;
            float2 _ForegroundSimulationTex_Size;

            float4 _Color;
            float3 _ForegroundAmbientColor;
            float _substrateDensity;
            float _particleAlignment;
            float _heightScale;
            float _exposure;
            float3 _whitePointLog;
            float3 _blackPointLog;
            float _ObscurityStrength;
            float _ObscurityVariation;


            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                
                o.vertex /= o.vertex.w;

                o.screen_uv = mul(_ForegroundSimulationUVTransform, float4(o.vertex.xy, 0, 1)).xy;
                o.screen_uv.y = 1+o.screen_uv.y;
                return o;
            }

            float4 frag (v2f i) : SV_Target0
            {
                ToneMappingShape tone_shape = {
                    _exposure,
                    _whitePointLog,
                    _blackPointLog
                };

                float4 c = tex2D(_MainTex, i.uv);

                float alpha = c.a * _Color.a;

                float obscurity = 1-tex2Dlod(_TransmissibilityTex, float4(i.screen_uv, 0, 0)).r;

               // float3 simulatedLight = float3(i.screen_uv, 0);
                float3 simulatedLight = tex2Dlod(_ForegroundSimulationTex, float4(i.screen_uv,0,_ForegroundSimulationLOD)).rgb;
                float3 ambientLight = tex2D(_ForegroundAmbientTex, i.uv).rgb * _ForegroundAmbientColor;
                float3 totalLight = simulatedLight*(1 + _ObscurityStrength*(pow(obscurity,_ObscurityVariation) - 1)) + ambientLight;

                return float4(ToneMap_UE5(c.rgb * _Color.rgb * totalLight, tone_shape),1) * alpha;
            }
            ENDCG
        }
    }

}