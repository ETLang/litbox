using System;
using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class SimulationTexturePicker : LitboxComponent {
    public enum TextureType {
        HDR,
        Variance,
        Importance,
        ForwardAccumulation,
        AI_ToneMapped,
        AI_HDR,
        Albedo,
        Transmissibility,
        NormalRoughness,
        QuadTree,
        AnalysisA,
        AnalysisB,
        AnalysisC,
    }

    [SerializeField] public Simulation simulation;
    //[SerializeField] private AIAccelerator aiAccelerator;
    [SerializeField] public TextureType type = TextureType.HDR;

    int _analysisAKernelId;
    int _analysisBKernelId;
    int _analysisCKernelId;

    ComputeShader _analysisShader;
    RenderTexture _analysisTargetA;
    RenderTexture _analysisTargetB;
    RenderTexture _analysisTargetC;

    void Awake()
    {
        _analysisShader = (ComputeShader)Resources.Load("Analysis");
        _analysisAKernelId = _analysisShader.FindKernel("AnalysisA");
        _analysisBKernelId = _analysisShader.FindKernel("AnalysisB");
        _analysisCKernelId = _analysisShader.FindKernel("AnalysisC");
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
        case TextureType.ForwardAccumulation:
            value = simulation?.TracerA is ITracerDebug debugA ? debugA.ForwardAccumulation : null;
            break;
        case TextureType.AnalysisA:
            ConfigureAnalysisA();
            value = _analysisTargetA;
            break;
        case TextureType.AnalysisB:
            ConfigureAnalysisB();
            value = _analysisTargetB;
            break;
        case TextureType.AnalysisC:
            ConfigureAnalysisC();
            value = _analysisTargetC;
            break;
        }

        if(value != null) {
            renderer.material.SetTexture("_MainTex", value);
        }

        if(!simulation.IsRunning) {
            MaybeRunAnalyses();
        }
    }

    void ConfigureAnalysisA()
    {
        if(_analysisTargetA != null) { return; }
        _analysisTargetA = this.CreateRWTexture(simulation.width, simulation.height, RenderTextureFormat.RFloat);
    }

    void ConfigureAnalysisB()
    {
        if(_analysisTargetB != null) { return; }
        _analysisTargetA = this.CreateRWTexture(simulation.width, simulation.height, RenderTextureFormat.RFloat);
        _analysisTargetB = this.CreateRWTexture(simulation.width, simulation.height, RenderTextureFormat.RFloat);
    }

    void ConfigureAnalysisC()
    {
        if(_analysisTargetC != null) { return; }
        _analysisTargetC = this.CreateRWTexture(simulation.width, simulation.height, RenderTextureFormat.RFloat);
    }

    private void OnSimulationUpdated(int frameCount)
    {
        MaybeRunAnalyses();
    }

    void MaybeRunAnalyses()
    {
        switch(type) {
        case TextureType.AnalysisA:
            // Compute full resolution relative variance
            RunAnalysis(_analysisAKernelId, _analysisTargetA);
            break;
        case TextureType.AnalysisB:
            // Perform median filter on AnalysisA to produce AnalysisB
            RunAnalysis(_analysisAKernelId, _analysisTargetA);
            RunAnalysis(_analysisBKernelId, _analysisTargetB, _analysisTargetA);
            break;
        case TextureType.AnalysisC:
            // Perform 5x5 adaptive filter on AnalysisB to produce AnalysisC
            RunAnalysis(_analysisAKernelId, _analysisTargetA);
            RunAnalysis(_analysisBKernelId, _analysisTargetB, _analysisTargetA);
            RunAnalysis(_analysisCKernelId, _analysisTargetC, _analysisTargetB);
            break;
        }
    }

    private void RunAnalysis(int kernel, RenderTexture target, RenderTexture previous = null)
    {
        if(kernel == -1) { return; }
        if(simulation == null) { return; }
        if(target == null) { return; }

        var args = GetComponent<AnalysisParameters>();

        if(args == null) {
            Debug.LogError("AnalysisParameters component is required for analysis compute shader.");
            return;
        }

        _analysisShader.SetFloat("_sigma_spatial", args.SigmaSpatial);
        _analysisShader.SetFloat("_sigma_albedo", args.SigmaAlbedo);
        _analysisShader.SetFloat("_sigma_luminance_tight", args.SigmaLuminanceTight);
        _analysisShader.SetFloat("_sigma_luminance_loose", args.SigmaLuminanceLoose);
        _analysisShader.SetFloat("_k_luminance", args.KLuminance);

        _analysisShader.RunKernel(kernel, target.width, target.height,
            ("_out_analysis", target),
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
            ("_in_previous_analysis", previous)
        );
    }
}