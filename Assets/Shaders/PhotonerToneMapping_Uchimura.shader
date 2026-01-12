Shader "Hidden/PhotonerToneMapping_Uchimura"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            #include "ToneMapping.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            float _Exposure;
            float _Contrast;
            float _LinearBase;
            float _LinearSpan;
            float3 _BlackTightness;
            float3 _BlackPedestal;
            float _MaximumBrightness;

            fixed4 frag (v2f i) : SV_Target
            {
                float4 hdrColor = tex2D(_MainTex, i.uv);

                ToneMappingShape_Uchimura tone_shape = {
                    _Contrast,
                    _LinearBase,
                    _LinearSpan,
                    _BlackTightness,
                    _BlackPedestal,
                    _MaximumBrightness
                };
                
                return float4(ToneMap_Uchimura(hdrColor.rgb * _Exposure, tone_shape), 1);
            }
            ENDCG
        }
    }
}