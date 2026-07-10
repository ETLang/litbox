Shader "Portfolio/PortfolioSpriteShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _LightMap ("Lightmap", 2D) = "black" {}
        _Color ("Color", Color) = (1,1,1,1)
        _LightMod ("Light Mod", Color) = (1,1,1,1)
        _LightDetail ("Light Detail", Float) = 0
        _Ambience ("Ambient", Color) = (0,0,0,0)
        _Emissive ("Emissive", Color) = (0,0,0,0)
        _LightingUVTransform("Sim UV Transform", Vector) = (1,1,0,0)
        _Metallic("Metallic", Range(0,1)) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        ZTest Off
        ZWrite Off


        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
                float2 lightUV : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            sampler2D _LightMap;
            float4 _LightMap_ST;
            fixed4 _Color;
            fixed4 _LightMod;
            float _LightDetail;
            fixed4 _Ambience;
            fixed4 _Emissive;
            float4 _LightingUVTransform;
            float _Metallic;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                float2 screenPos = o.vertex.xy / o.vertex.w;
                o.lightUV = screenPos.xy * _LightingUVTransform.xy + _LightingUVTransform.zw;
                
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv) * _Color;

                // sample lightmap
                fixed4 light = tex2Dlod(_LightMap, float4(i.lightUV, 0, _LightDetail));
                light = light * _LightMod + _Ambience;

                // combine
                col *= light;
                col += _Emissive;

                col.rgb *= col.a;

                return col;
            }
            ENDCG
        }
    }
}
