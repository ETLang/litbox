using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class Denoiser3 : LitboxComponent
{
    private Simulation _simulation;
    private ComputeShader _denoiserShader;
    private int _denoiseKernel;

    public RenderTexture DenoisedOutput { get; private set; }

    private int _currentWidth = -1;
    private int _currentHeight = -1;

    // Wrapped at a small bound (matches SimulationResources.frameIndex on the WebGPU side) so the
    // per-pixel seed jitter decorrelates frame-to-frame without approaching f32's exact-integer
    // precision cliff - see the shader's own doc comment on _FrameIndex's use.
    private int _frameIndex = 0;

    public float normalSensitivity = 8.0f;
    public float albedoSensitivity = 4.0f;
    public float densitySensitivity = 1.0f;
    public float varianceScale = 4;
    public float darknessNoiseFloor = 0.002f;

    [Tooltip("Distance-bias split cutoff, in node-relative texels - see the shader's ShouldSplit doc comment.")]
    public float maxSplitDistance = 2.0f;

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

        TracerPostProcessor.Instance.GenerateDenoisingFilterQuadtree(simulation.GBuffer.AlbedoAlpha, simulation.GBuffer.NormalRoughness, simulation.GBuffer.Density, source, quadtree);

        // The guided blur below writes to a scratch buffer, not directly to `output` - the guided
        // post-filter (DitherFilter.compute) always runs immediately after, over this not-yet-final
        // result, and its own output is what actually belongs in `output`. A neighbor-gathering
        // filter can't run in place, hence the separate buffer.
        var denoiseScratch = BufferManager.AcquireTexture(output.width, output.height, output.format);

        _denoiserShader.SetTexture(_denoiseKernel, "_Input", source);
        _denoiserShader.SetTexture(_denoiseKernel, "_Variance", simulation.VarianceMap);
        _denoiserShader.SetTexture(_denoiseKernel, "_Albedo", simulation.GBuffer.AlbedoAlpha);
        _denoiserShader.SetTexture(_denoiseKernel, "_NormalRoughness", simulation.GBuffer.NormalRoughness);
        _denoiserShader.SetTexture(_denoiseKernel, "_Density", simulation.GBuffer.Density);
        _denoiserShader.SetTexture(_denoiseKernel, "_Output", denoiseScratch);
        _denoiserShader.SetTexture(_denoiseKernel, "_Quadtree", quadtree);

        _denoiserShader.SetVector("_InputSize", new Vector4(simulation.width, simulation.height, 1.0f / simulation.width, 1.0f / simulation.height));

        _denoiserShader.SetFloat("_NormalSensitivity", normalSensitivity);
        _denoiserShader.SetFloat("_AlbedoSensitivity", albedoSensitivity);
        _denoiserShader.SetFloat("_DensitySensitivity", densitySensitivity);
        _denoiserShader.SetFloat("_VarianceScale", varianceScale);
        _denoiserShader.SetFloat("_DarknessNoiseFloor", darknessNoiseFloor);
        _denoiserShader.SetFloat("_MaxLOD", maxLOD);
        _denoiserShader.SetFloat("_MaxSplitDistance", maxSplitDistance);

        // Reuse TracerPostProcessor's validated adaptive-sigma pattern verbatim (already the single
        // source of truth for these three, used by FilterVariance) rather than duplicating tuned
        // constants here.
        _denoiserShader.SetFloat("_SigmaLuminanceTight", TracerPostProcessor.Instance.SigmaLuminanceTight);
        _denoiserShader.SetFloat("_SigmaLuminanceLoose", TracerPostProcessor.Instance.SigmaLuminanceLoose);
        _denoiserShader.SetFloat("_KLuminance", TracerPostProcessor.Instance.KLuminance);

        _frameIndex = (_frameIndex + 1) % 4096;
        _denoiserShader.SetFloat("_FrameIndex", _frameIndex);

        _denoiserShader.SetShaderFlag("APPLY_RADIANCE", applyRadiance);

        _denoiserShader.Dispatch(_denoiseKernel, (simulation.width + 7) / 8, (simulation.height + 7) / 8, 1);

        TracerPostProcessor.Instance.ApplyDitherFilter(denoiseScratch, simulation.GBuffer.AlbedoAlpha, simulation.GBuffer.NormalRoughness, simulation.GBuffer.Density, output);

        BufferManager.Release(ref quadtree);
        BufferManager.Release(ref denoiseScratch);
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