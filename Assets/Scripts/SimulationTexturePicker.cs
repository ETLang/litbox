using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class SimulationTexturePicker : MonoBehaviour {
    public enum TextureType {
        Photons_Forward,
        Photons_Raw,
        Camera_Accum,
        HDR,
        Variance,
        AI_ToneMapped,
        AI_HDR,
        Albedo,
        Transmissibility,
        NormalRoughness,
        QuadTree,
    }

    [SerializeField] public Simulation simulation;
    //[SerializeField] private AIAccelerator aiAccelerator;
    [SerializeField] public TextureType type = TextureType.HDR;

    void OnDisable()
    {
        GetComponent<Renderer>().material.SetTexture("_MainTex", null);    
    }

    void LateUpdate()
    {
        if(!simulation) return;

        var renderer = GetComponent<Renderer>();
        Texture value = null;

        switch(type) {
        // case TextureType.Photons_Forward:
        //     value = simulation?.SimulationForwardHDR;
        //     break;
        // case TextureType.Camera_Accum:
        //     value = simulation?.SimulationBackwardAccumulated;
        //     break;
        // case TextureType.Photons_Raw:
        //     value = simulation?.SimulationPhotonsRaw;
        //     break;
        case TextureType.HDR:
            value = simulation?.SimulationOutputHDR;
            break;
        case TextureType.AI_ToneMapped:
            //value = aiAccelerator?.ToneMappedOutputTexture;
            break;
        case TextureType.AI_HDR:
            //value = aiAccelerator?.HDROutputTexture;
            break;
        case TextureType.Albedo:
            value = simulation?.GBuffer.AlbedoAlpha;
            break;
        case TextureType.Transmissibility:
            value = simulation?.GBuffer.Transmissibility;
            break;
        case TextureType.NormalRoughness:
            value = simulation?.GBuffer.NormalRoughness;
            break;
        case TextureType.QuadTree:
            value = simulation?.GBuffer.QuadTreeLeaves;
            break;
        case TextureType.Variance:
            value = simulation?.VarianceMap;
            break;
        }

        if(value != null) {
            renderer.material.SetTexture("_MainTex", value);
        }
    }
}