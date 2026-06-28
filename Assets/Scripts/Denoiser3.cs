using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class Denoiser3 : LitboxComponent
{
    private Simulation _simulation;
    private ComputeShader _denoiserShader;
    private int _denoiseKernel;

    public RenderTexture DenoisedOutput { get; private set; }

// float4 _InputSize; // x: Width, y: Height, z: 1/Width, w: 1/Height

// float _NormalSensitivity;    // Tolerance for normal differences (e.g., 8.0)
// float _AlbedoSensitivity;    // Tolerance for albedo differences (e.g., 4.0)
// float _VarianceScale;        // Maps variance to initial starting MIP depth
// float _DarknessNoiseFloor;   // Low-light floor
// float _MaxLOD;          // Deepest allowed mip level (e.g., 5.0 or 6.0)
// float _SplitThreshold;       // Structural similarity weight below which we split a pixel (e.g., 0.3)

    private int _currentWidth = -1;
    private int _currentHeight = -1;

    public float normalSensitivity = 8.0f;
    public float albedoSensitivity = 4.0f;
    public float transmissibilitySensitivity = 1.0f;
    public float varianceSensitivity = 10.0f;
    public float varianceScale = 4;
    public float darknessNoiseFloor = 0.002f;
    [Range(0,1)] public float splitThreshold = 0.3f;

    [Tooltip("Maximum LOD to sample from the input texture")]
    [Range(0, 6)]
    public float maxLOD = 4.0f;

    void OnEnable()
    {
        if (_simulation == null) _simulation = GetComponent<Simulation>();

        _denoiserShader = (ComputeShader)Resources.Load("Denoiser3");
        if (_denoiserShader == null)
        {
            Debug.LogError("Denoiser2 compute shader not found in Resources folder.");
            enabled = false;
            return;
        }

        _denoiseKernel = _denoiserShader.FindKernel("CSMain");
        _simulation.OnPostProcess += OnSimulationPostProcess;
    }

    protected override void OnDisable()
    {
        if (_simulation != null)
        {
            _simulation.OnPostProcess -= OnSimulationPostProcess;
        }

        ReleaseOutput();
    }

    private void ReleaseOutput()
    {
        _currentWidth = -1;
        _currentHeight = -1;

        if (DenoisedOutput != null)
        {
            DisposeOnNextFrame(() => DestroyImmediate(DenoisedOutput));
            DenoisedOutput = null;
        }
    }

    public void Denoise(Simulation simulation, RenderTexture source, RenderTexture output, bool applyRadiance)
    {
        var quadtree = BufferManager.AcquireTexture(simulation.width / 2, simulation.height / 2, RenderTextureFormat.RFloat, true);

        TracerPostProcessor.Instance.GenerateDenoisingFilterQuadtree(simulation.GBuffer.AlbedoAlpha, simulation.GBuffer.NormalRoughness, simulation.GBuffer.Transmissibility, source, quadtree);

        _denoiserShader.SetTexture(_denoiseKernel, "_Input", source);
        _denoiserShader.SetTexture(_denoiseKernel, "_Variance", simulation.VarianceMap);
        _denoiserShader.SetTexture(_denoiseKernel, "_Albedo", simulation.GBuffer.AlbedoAlpha);
        _denoiserShader.SetTexture(_denoiseKernel, "_NormalRoughness", simulation.GBuffer.NormalRoughness);
        _denoiserShader.SetTexture(_denoiseKernel, "_Transmissibility", simulation.GBuffer.Transmissibility);
        _denoiserShader.SetTexture(_denoiseKernel, "_Output", output);
        _denoiserShader.SetTexture(_denoiseKernel, "_Quadtree", quadtree);

        _denoiserShader.SetVector("_InputSize", new Vector4(simulation.width, simulation.height, 1.0f / simulation.width, 1.0f / simulation.height));

        _denoiserShader.SetFloat("_NormalSensitivity", normalSensitivity);
        _denoiserShader.SetFloat("_AlbedoSensitivity", albedoSensitivity);
        _denoiserShader.SetFloat("_TransmissibilitySensitivity", transmissibilitySensitivity);
        _denoiserShader.SetFloat("_VarianceSensitivity", varianceSensitivity);
        _denoiserShader.SetFloat("_VarianceScale", varianceScale);
        _denoiserShader.SetFloat("_DarknessNoiseFloor", darknessNoiseFloor);
        _denoiserShader.SetFloat("_SplitThreshold", splitThreshold);
        _denoiserShader.SetFloat("_MaxLOD", maxLOD);

        _denoiserShader.SetShaderFlag("APPLY_RADIANCE", applyRadiance);

        _denoiserShader.Dispatch(_denoiseKernel, (simulation.width + 7) / 8, (simulation.height + 7) / 8, 1);

        BufferManager.Release(ref quadtree);
    }

    private RenderTexture OnSimulationPostProcess(RenderTexture source)
    {
        if (_simulation.denoiser == this) return source;

        if (_simulation.width != _currentWidth || _simulation.height != _currentHeight)
        {
            ReleaseOutput();
            
            _currentWidth = _simulation.width;
            _currentHeight = _simulation.height;

            DenoisedOutput = this.CreateRWTextureWithMips(_currentWidth, _currentHeight, RenderTextureFormat.ARGBFloat);
            DenoisedOutput.name = "Denoiser2_ProceduralOutput";
        }

        Denoise(_simulation, source, DenoisedOutput, false);

        return DenoisedOutput;
    }
}