using System;
using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class SimulationTexturePicker : PhotonerComponent {
    [SerializeField] float brightnessThreshold = 1;
    [SerializeField] float varianceThreshold = 1;

    public enum TextureType {
        HDR,
        Variance,
        Importance,
        AI_ToneMapped,
        AI_HDR,
        Albedo,
        Transmissibility,
        NormalRoughness,
        QuadTree,
        AnalysisA,
    }

    [SerializeField] public Simulation simulation;
    //[SerializeField] private AIAccelerator aiAccelerator;
    [SerializeField] public TextureType type = TextureType.HDR;

    int _analysisAKernelId;

    ComputeShader _analysisShader;
    RenderTexture _analysisTarget;
    int _selectedAnalysisKernel = -1;

    void Awake()
    {
        _analysisShader = (ComputeShader)Resources.Load("Analysis");
        _analysisAKernelId = _analysisShader.FindKernel("AnalysisA");
    }

    void OnEnable()
    {
        if(simulation)
        {
            simulation.OnStep += OnSimulationUpdated;
        }
    }

    protected override void OnDisable()
    {
        GetComponent<Renderer>().material.SetTexture("_MainTex", null);    

        if(simulation)
        {
            simulation.OnStep -= OnSimulationUpdated;
        }
    }

    void LateUpdate()
    {
        if(!simulation) return;

        var renderer = GetComponent<Renderer>();
        Texture value = null;

        switch(type) {
        case TextureType.HDR:
            value = simulation?.SimulationOutputHDR;
            break;
        // case TextureType.AI_ToneMapped:
        //     //value = aiAccelerator?.ToneMappedOutputTexture;
        //     break;
        // case TextureType.AI_HDR:
        //     //value = aiAccelerator?.HDROutputTexture;
        //     break;
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
        case TextureType.Importance:
            value = simulation?.ImportanceMap;
            break;
        case TextureType.AnalysisA:
            ConfigureAnalysisA();
            value = _analysisTarget;
            break;
        }

        if(value != _analysisTarget) {
            _selectedAnalysisKernel = -1;
        }

        if(value != null) {
            renderer.material.SetTexture("_MainTex", value);
        }
    }

    void ConfigureAnalysisA()
    {
        if(_selectedAnalysisKernel == _analysisAKernelId) { return; }

        _selectedAnalysisKernel = _analysisAKernelId;
        _analysisTarget = this.CreateRWTexture(simulation.width, simulation.height, RenderTextureFormat.ARGBFloat);
    }

    private void OnSimulationUpdated(int frameCount)
    {
        if(_selectedAnalysisKernel == -1) { return; }

        _analysisShader.RunKernel(_selectedAnalysisKernel, _analysisTarget.width, _analysisTarget.height,
            ("_out_analysis", _analysisTarget),
            ("_in_albedo", simulation.GBuffer.AlbedoAlpha),
            ("_in_transmissibility", simulation.GBuffer.Transmissibility),
            ("_in_normal_roughness", simulation.GBuffer.NormalRoughness),
            ("_in_hdr_forward_a", simulation.TracerA.EarlyRadianceForImportanceSampling),
            ("_in_hdr_forward_b", simulation.TracerB.EarlyRadianceForImportanceSampling),
            ("_in_hdr_a", simulation.TracerA.TracerOutput),
            ("_in_hdr_b", simulation.TracerB.TracerOutput),
            ("_in_hdr_final", simulation.SimulationOutputHDR),
            ("_in_importance", simulation.ImportanceMap),
            ("_in_variance", simulation.VarianceMap),
            ("_brightness_threshold", brightnessThreshold),
            ("_variance_threshold", varianceThreshold)
        );
    }
}