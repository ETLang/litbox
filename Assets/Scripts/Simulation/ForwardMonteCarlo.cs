using System;
using UnityEngine;

public class ForwardMonteCarlo : Disposable
{
    #region GBuffer
    public PhotonerGBuffer GBuffer
    {
        get => _gBuffer;
        set
        {
            _gBuffer = value;

            BufferManager.Release(ref _rawPhotonBuffer);
            BufferManager.Release(ref _outputImageHDR);

            if(_gBuffer.AlbedoAlpha != null) {
                int w = _gBuffer.AlbedoAlpha.width;
                int h = _gBuffer.AlbedoAlpha.height;

                _rawPhotonBuffer = BufferManager.AcquireTexture(w * 3, h, RenderTextureFormat.RInt);
                _outputImageHDR = BufferManager.AcquireTexture(w, h, RenderTextureFormat.ARGBFloat, true);
                UpdateIntegrationInterval();
            }
        }
    }
    private PhotonerGBuffer _gBuffer = new PhotonerGBuffer();
    #endregion

    public RenderTexture RawPhotonBuffer => _rawPhotonBuffer;
    private RenderTexture _rawPhotonBuffer;

    public RenderTexture OutputImageHDR => _outputImageHDR;
    private RenderTexture _outputImageHDR;

    #region ImportanceSamplingTarget
    public Vector2 ImportanceSamplingTarget
    {
        get => _importanceSamplingTarget;
        set
        {
            _importanceSamplingTarget = value;

            if(GBuffer.AlbedoAlpha) {
                _forwardIntegrationShader.SetVector("g_importance_sampling_target", value * new Vector2(GBuffer.AlbedoAlpha.width, GBuffer.AlbedoAlpha.height));
            }
        }
    }
    private Vector2 _importanceSamplingTarget = new Vector2(0.5f, 0.5f);
    #endregion

    #region DisableBilinearWrites
    public bool DisableBilinearWrites
    {
        get => _disableBilinearWrites;
        set
        {
            _disableBilinearWrites = value;
            _forwardIntegrationShader.SetShaderFlag("BILINEAR_PHOTON_DISTRIBUTION", !_disableBilinearWrites);
        }
    }
    private bool _disableBilinearWrites;
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
        if(_gBuffer.AlbedoAlpha == null) return;

        _forwardIntegrationShader.SetFloat("g_integration_interval",
            Mathf.Max(0.01f, IntegrationInterval * _gBuffer.AlbedoAlpha.height));
    }
    #endregion

    public Matrix4x4 WorldToTargetTransform { get; set; } = Matrix4x4.identity;
    public int IterationsSinceClear { get; set; }
    public uint? OverrideBounceCount { get; set; }
    public int RaysToEmit { get; set; } = 65536;


    private ComputeShader _forwardIntegrationShader;

    public ForwardMonteCarlo()
    {
        _forwardIntegrationShader = (ComputeShader)Resources.Load("ForwardMonteCarlo");
        _forwardIntegrationShader.SetVector("g_importance_sampling_target", ImportanceSamplingTarget);
        _forwardIntegrationShader.SetShaderFlag("BILINEAR_PHOTON_DISTRIBUTION", !_disableBilinearWrites);
    }

    public void Clear()
    {
        RawPhotonBuffer.Clear(Color.clear);
    }

    private static Vector4 _LuminanceWeight = new Vector4(0.2126f, 0.7152f, 0.0722f, 0);
    private static float Luminance(Vector4 energy)
    {
        return Vector4.Dot(_LuminanceWeight, energy);
    }

    public void Integrate(params RTLightSource[] lights)
    {
        if(_gBuffer.AlbedoAlpha == null)
        {
            Debug.LogError("Integrate() called with no GBuffer set");
            return;
        }

        int width = RawPhotonBuffer.width;
        int height = RawPhotonBuffer.height;

        _forwardIntegrationShader.SetFloat("g_hdr_scale", (float)((double)uint.MaxValue * IterationsSinceClear / (width * height)));

        float totalLuma = 0;
        foreach (var light in lights)
        {
            totalLuma += Luminance(light.Energy);
        }

        if(totalLuma == 0) { return; }

        foreach (var light in lights)
        {
            int rays = (int)(Luminance(light.Energy) / totalLuma * RaysToEmit);
            SimulateLight(light, rays);
        }
        
        _forwardIntegrationShader.RunKernel("ConvertToHDR", width, height,
            ("g_output_raw", RawPhotonBuffer),
            ("g_output_hdr", OutputImageHDR));
    }

    void SimulateLight(RTLightSource light, int rays)
    {
        // Round up to the next multiple of 64 rays for good thread grouping
        rays = ((rays - 1) / 64 + 1) * 64;

        string simulateKernel = null;
        var lightToTargetSpace = WorldToTargetTransform * light.WorldTransform;
        double photonEnergy = (double)uint.MaxValue / rays;
        float emissionOutscatter = 0;
        uint bounces = OverrideBounceCount ?? light.bounces;

        string kernelFormat = "Simulate_{0}";

        switch (light)
        {
            case RTPointLight pt:
                simulateKernel = string.Format(kernelFormat, "PointLight");
                emissionOutscatter = pt.emissionOutscatter;
                break;
            case RTSpotLight _:
                simulateKernel = string.Format(kernelFormat, "SpotLight");
                break;
            case RTLaserLight _:
                simulateKernel = string.Format(kernelFormat, "LaserLight");
                break;
            case RTAmbientLight _:
                simulateKernel = string.Format(kernelFormat, "AmbientLight");
                break;
            case RTFieldLight field:
                simulateKernel = string.Format(kernelFormat, "FieldLight");
                _forwardIntegrationShader.SetTexture(_forwardIntegrationShader.FindKernel(simulateKernel), "g_lightFieldTexture", field.lightTexture ? field.lightTexture : Texture2D.whiteTexture);
                emissionOutscatter = field.emissionOutscatter;
                break;
            case RTDirectionalLight dir:
                simulateKernel = string.Format(kernelFormat, "DirectionalLight");
                _forwardIntegrationShader.SetVector("g_directionalLightDirection", lightToTargetSpace.MultiplyVector(new Vector3(0, -1, 0)));
                break;
        }

        float integrationInterval = Mathf.Max(1, IntegrationInterval * OutputImageHDR.height);
        float integrationIntervalEnergyAdjustment = 1.0f / integrationInterval;
        _forwardIntegrationShader.SetFloat("g_lightEmissionOutscatter", emissionOutscatter);
        _forwardIntegrationShader.SetVector("g_lightEnergy", light.Energy * (float)photonEnergy * integrationIntervalEnergyAdjustment);
        _forwardIntegrationShader.SetInt("g_bounces", (int)bounces);
        _forwardIntegrationShader.SetMatrix("g_lightToTarget", lightToTargetSpace.transpose);

        // TODO: Try making IntegrationInterval a pixel length, rather than a ratio.
        _forwardIntegrationShader.SetFloat("g_integration_interval", integrationInterval);

        _forwardIntegrationShader.RunKernel(simulateKernel, rays,
            ("g_rand", BufferManager.GetRandomSeedBuffer(rays)),
            ("g_output_raw", RawPhotonBuffer),
            ("g_albedo", _gBuffer.AlbedoAlpha),
            ("g_transmissibility", _gBuffer.Transmissibility),
            ("g_normalAlignment", _gBuffer.NormalRoughness),
            ("g_quadTreeLeaves", _gBuffer.QuadTreeLeaves),
            ("g_mieScatteringLUT", BufferManager.MieScatteringLUT),
            ("g_teardropScatteringLUT", BufferManager.TeardropScatteringLUT),
            ("g_bdrfLUT", BufferManager.BRDFLUT));
    }

    protected override void OnDispose()
    {
        base.OnDispose();

        BufferManager.Release(ref _rawPhotonBuffer);
        BufferManager.Release(ref _outputImageHDR);
    }
}