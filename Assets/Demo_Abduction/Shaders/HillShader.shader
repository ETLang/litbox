Shader "Unlit/HillShader"
{
    Properties
    {
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            Blend Off

            CGPROGRAM
            #pragma vertex hill_vert
            #pragma fragment hill_frag
            #include "HillShader.cginc"
            ENDCG
        }

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            ZTest Equal
            ZWrite Off

            CGPROGRAM
            #pragma vertex hill_vert
            #pragma fragment hill_frag
            #include "HillShader.cginc"
            ENDCG
        }
    }
}
