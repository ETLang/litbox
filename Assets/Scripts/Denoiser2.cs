using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class Denoiser2 : LitboxComponent
{
    private Simulation _simulation;
    private ComputeShader _denoiserShader;
    private int _denoiseKernel;

    public RenderTexture DenoisedOutput { get; private set; }

    private int _currentWidth = -1;
    private int _currentHeight = -1;

    public float varianceThresholdMin = 0.002f;
    public float varianceThresholdMax = 0.05f;
    public float luminanceThresholdMin = 0;
    public float luminanceThresholdMax = 0.002f;

    [Tooltip("Maximum LOD to sample from the input texture")]
    [Range(0, 6)]
    public float maxLOD = 6.0f;

    void OnEnable()
    {
        if (_simulation == null) _simulation = GetComponent<Simulation>();

        _denoiserShader = (ComputeShader)Resources.Load("Denoiser2");
        if (_denoiserShader == null)
        {
            Debug.LogError("Denoiser2 compute shader not found in Resources folder.");
            enabled = false;
            return;
        }

        _denoiseKernel = _denoiserShader.FindKernel("ProceduralDenoise");
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

    private RenderTexture OnSimulationPostProcess(RenderTexture source)
    {
        if (_simulation.width != _currentWidth || _simulation.height != _currentHeight)
        {
            ReleaseOutput();
            
            _currentWidth = _simulation.width;
            _currentHeight = _simulation.height;
            
            var desc = new RenderTextureDescriptor(_currentWidth, _currentHeight, RenderTextureFormat.ARGBFloat, 0)
            {
                enableRandomWrite = true,
                autoGenerateMips = false
            };
            DenoisedOutput = new RenderTexture(desc);
            DenoisedOutput.name = "Denoiser2_ProceduralOutput";
            DenoisedOutput.Create();
        }

        _denoiserShader.SetTexture(_denoiseKernel, "_Input", source);
        _denoiserShader.SetTexture(_denoiseKernel, "_Variance", _simulation.VarianceMap);
        _denoiserShader.SetTexture(_denoiseKernel, "_Albedo", _simulation.GBuffer.AlbedoAlpha);
        _denoiserShader.SetTexture(_denoiseKernel, "_NormalRoughness", _simulation.GBuffer.NormalRoughness);
        _denoiserShader.SetTexture(_denoiseKernel, "_Density", _simulation.GBuffer.Density);
        _denoiserShader.SetTexture(_denoiseKernel, "_Output", DenoisedOutput);

        _denoiserShader.SetFloat("_VarianceThresholdMin", varianceThresholdMin);
        _denoiserShader.SetFloat("_VarianceThresholdMax", varianceThresholdMax);
        _denoiserShader.SetFloat("_LuminanceThresholdMin", luminanceThresholdMin);
        _denoiserShader.SetFloat("_LuminanceThresholdMax", luminanceThresholdMax);
        _denoiserShader.SetFloat("_MaxLOD", maxLOD);

        _denoiserShader.Dispatch(_denoiseKernel, (_currentWidth + 7) / 8, (_currentHeight + 7) / 8, 1);

        return DenoisedOutput;
    }
}