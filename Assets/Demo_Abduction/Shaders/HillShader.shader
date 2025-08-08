Shader "Unlit/HillShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "green" {}
        _LeftHeight ("Left Height", Float) = 5.0
        _PeakHeight ("Peak Height", Float) = 7.0
        _RightHeight ("Right Height", Float) = 5.0
        _Pinch ("Pinch", Float) = 0.0
        _BaseColor ("Base Color", Color) = (0.1, 0.3, 0.13, 1)
        _HighlightColor ("Highlight Color", Color) = (0.7, 0.9, 0.8, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

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
                float3 normal : NORMAL;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _LeftHeight;
            float _PeakHeight;
            float _RightHeight;
            float _Pinch;
            float4 _BaseColor;
            float4 _HighlightColor;

            v2f vert (appdata v)
            {
                const float pi = 3.141592564f;
                const float xLeft = -5;
                const float xRight = 5;

                const float gaussianLimit = 2;
                const float gaussianLowerbound = exp(-pow(gaussianLimit, 2));


                // float x = lerp(xLeft, xRight, lerp(v.uv.x, 2*v.uv.x - w, _Pinch * pow(v.uv.y, 3)));
                // float w = (1 - cos(v.uv.x * pi)) / 2.0f;
                // float top = lerp(_LeftHeight, _RightHeight, w);
                // float y = top * v.uv.y;

                float x = lerp(xLeft, xRight, v.uv.x);
                float wx = lerp(-gaussianLimit, gaussianLimit, v.uv.x);
                float w = exp(-pow(wx, 2));
                float dy = -2*wx*w;
                w = (w - gaussianLowerbound) / (1 - gaussianLowerbound);
                float top;

                if(v.uv.x < 0.5) {
                    top = lerp(_LeftHeight, _PeakHeight, w);
                    dy *= (_PeakHeight - _LeftHeight);
                } else {
                    top = lerp(_RightHeight, _PeakHeight, w);
                    dy *= (_PeakHeight - _RightHeight);
                }
                float y = top * v.uv.y;

                float3 n;
                if(abs(dy) < 1e-6)
                    n = float3(0,1,0);
                else 
                    n = normalize(float3((2*gaussianLimit)/(xRight-xLeft), -1/dy, 0)) * sign(-dy);
                n = lerp(normalize(float3(0,1,2)), n, pow(v.uv.y, 2));

                v2f o;
                o.vertex = UnityObjectToClipPos(float4(x, y, 0, 1));
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.normal = n;
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float2 grid = saturate(25 * abs(frac(i.uv * 20) * 2 - 1));
                float4 grid_c = lerp(float4(1,0,0,1), float4(1,1,1,1), min(grid.x, grid.y));

                float4 normal_c = float4(abs(i.normal.x), i.normal.y, 0, 1);

                float3 l = float3(0,1,0);
                float4 faux_light_c = lerp(_BaseColor, _HighlightColor, pow(saturate(dot(l, i.normal)), 3));
                //fixed4 col = tex2D(_MainTex, i.uv);

                return faux_light_c;
                //return normal_c;
                //return grid_c;
            }
            ENDCG
        }
    }
}
