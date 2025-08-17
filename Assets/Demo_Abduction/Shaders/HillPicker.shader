Shader "Hidden/HillPicker"
{
    Properties
    {
        _Color ("Main Color", Color) = (1,1,1,1)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex hill_vert
            #pragma fragment picker_frag
            #include "HillShader.cginc"
            float4 _Color;
            fixed4 picker_frag (v2f i) : SV_Target
            {
                return _Color;
            }
            ENDCG
        }
    }
}