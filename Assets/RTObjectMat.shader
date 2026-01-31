Shader "RT/Object"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Color ("Color", Color) = (1,1,1,1) 
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            Blend 0 One OneMinusSrcAlpha
            Blend 1 Zero SrcColor
            Blend 2 One Zero

            ZWrite Off

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

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
            };

            struct gbuffer_output
            {
                float4 albedo : SV_Target0;
                float4 transmissibility: SV_Target1;
                float4 normal : SV_Target2;
            };

            float4 _MainTex_ST;
            sampler2D _MainTex;
            sampler2D _NormalTex;

            float4 _Color;
            float _substrateDensity;
            float _particleAlignment;
            float _heightScale;


            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.normal = UnityObjectToWorldNormal(v.normal);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            gbuffer_output frag (v2f i)
            {
                gbuffer_output output;

                float4 c = tex2D(_MainTex, i.uv);
                float4 n = tex2D(_NormalTex, i.uv);

                float imageDensity = _substrateDensity * c.a;
                float imageTransmissibility = 1 - imageDensity;

                float t = pow(imageTransmissibility, 100.0/_ScreenParams.y);

                output.albedo = float4(c.rgb * _Color.rgb,1) * c.a * _Color.a;
                output.transmissibility = float4(t,t,0,1);
                output.normal = float4(i.normal, _particleAlignment);
                return output;
            }
            ENDCG
        }
    }
}
