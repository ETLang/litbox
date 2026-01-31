using Unity.Mathematics;
using UnityEngine;

public class BackwardMonteCarlo : Disposable
{
    public PhotonerGBuffer GBuffer { get; set; }

    #region InputImage
    public RenderTexture InputImage
    {
        get => _inputImage;
        set
        {
            _inputImage = value;

            BufferManager.Release(ref _outputImage);
            BufferManager.Release(ref _accumulationImage);

            if(_inputImage != null) {
                int w = _inputImage.width;
                int h = _inputImage.height;

                _accumulationImage = BufferManager.AcquireTexture(w, h, RenderTextureFormat.ARGBFloat);
                _outputImage = BufferManager.AcquireTexture(w, h, _inputImage.format, true);
                UpdateIntegrationInterval();
            }
        }
    }
    private RenderTexture _inputImage;
    #endregion

    public RenderTexture AccumulationImage => _accumulationImage;
    private RenderTexture _accumulationImage;

    public RenderTexture OutputImage => _outputImage; 
    private RenderTexture _outputImage;

    public RenderTexture ImportanceMap { get; set; }

    #region ImportanceSamplingTarget
    public Vector2 ImportanceSamplingTarget
    {
        get => _importanceSamplingTarget;
        set
        {
            _importanceSamplingTarget = value;

            if(GBuffer.AlbedoAlpha) {
                _backwardIntegrationShader.SetVector("g_importance_sampling_target", value * new Vector2(GBuffer.AlbedoAlpha.width, GBuffer.AlbedoAlpha.height));
            }
        }
    }
    private Vector2 _importanceSamplingTarget = new Vector2(0.5f, 0.5f);
    #endregion

    #region IntegrationInterval
    public float IntegrationInterval
    {
        get => _integrationInterval;
        set
        {
            _integrationInterval = value;
            UpdateIntegrationInterval();
        }
    }
    private float _integrationInterval = 0.2f;

    private void UpdateIntegrationInterval()
    {
        if(InputImage == null) return;

        _backwardIntegrationShader.SetFloat("g_integration_interval",
            Mathf.Max(0.01f, IntegrationInterval * InputImage.height));
    }
    #endregion

    private ComputeShader _backwardIntegrationShader;
    private int _frameCount;

    public BackwardMonteCarlo()
    {
        _backwardIntegrationShader = (ComputeShader)Resources.Load("BackwardMonteCarlo");
    }

    public void Clear()
    {
        AccumulationImage.Clear(Color.clear);
        _frameCount = 0;
    }

    public void Integrate()
    {
        if(InputImage == null) return;
        if(ImportanceMap == null) return;
        
        var albedo = (Texture)GBuffer.AlbedoAlpha ?? Texture2D.whiteTexture;
        var transmissibility = (Texture)GBuffer.Transmissibility ?? Texture2D.whiteTexture;

        int width = InputImage.width;
        int height = InputImage.height;
        _frameCount++;

        _backwardIntegrationShader.RunKernel("Simulate_Camera", width, height,
            ("g_rand", BufferManager.GetRandomSeedBuffer(width * height)),
            ("g_hdr", InputImage),
            ("g_output_hdr", AccumulationImage),
            ("g_albedo", albedo),
            ("g_transmissibility", transmissibility),
            ("g_importanceMap", ImportanceMap),
            ("g_mieScatteringLUT", BufferManager.MieScatteringLUT),
            ("g_teardropScatteringLUT", BufferManager.TeardropScatteringLUT),
            ("g_bdrfLUT", BufferManager.BRDFLUT));

        // NORMALIZATION
        _backwardIntegrationShader.RunKernel("Camera_Buffer_Divide", width, height,
            ("g_hdr", AccumulationImage),
            ("g_output_hdr", OutputImage),
            ("g_count", _frameCount),
            ("g_frame_count", _frameCount));
    }
}