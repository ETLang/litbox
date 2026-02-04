using UnityEngine;

public class LightTransportTracer : Disposable, ITracer
{
    private ForwardMonteCarlo _forwardIntegrator;
    long _lastForwardWriteCount = 0;
    float _lastPerformanceUpdateTime = 0;

    public RenderTexture TracerOutput => _forwardIntegrator.OutputImageHDR;
    RenderTexture ITracer.EarlyRadianceForImportanceSampling => null;

    public PhotonerGBuffer GBuffer
    {
        get => _forwardIntegrator.GBuffer;
        set => _forwardIntegrator.GBuffer = value;
    }

    public bool DisableBilinearWrites
    {
        get => _forwardIntegrator.DisableBilinearWrites;
        set => _forwardIntegrator.DisableBilinearWrites = value;
    }

    public Matrix4x4 WorldToTargetTransform
    {
        get => _forwardIntegrator.WorldToTargetTransform;
        set => _forwardIntegrator.WorldToTargetTransform = value;
    }

    public float IntegrationInterval
    {
        get => _forwardIntegrator.IntegrationInterval;
        set => _forwardIntegrator.IntegrationInterval = value;
    }

    public uint? OverrideBounceCount
    {
        get => _forwardIntegrator.OverrideBounceCount;
        set => _forwardIntegrator.OverrideBounceCount = value;
    }

    public int RaysToEmit
    {
        get => _forwardIntegrator.RaysToEmit;
        set => _forwardIntegrator.RaysToEmit = value;
    }

    public long ForwardWritesPerSecond { get; private set; }
    public long BackwardReadsPerSecond => 0;

    public LightTransportTracer()
    {
        _forwardIntegrator = new ForwardMonteCarlo() { FinalizeOutscatterDensity = true };
        AutoDispose(_forwardIntegrator);
    }

    // TODO Somewhere in here belongs importance sampling logic.

    public void NewScene()
    {
        _forwardIntegrator.IterationsSinceClear = 0;
        _forwardIntegrator.Clear();
    }

    public void BeginTrace(params RTLightSource[] lights)
    {
        _forwardIntegrator.IterationsSinceClear++;
        _forwardIntegrator.Integrate(lights);
    }

    public void EndTrace(RenderTexture importanceMap)
    {
    }

    public async void UpdatePerformanceMetrics()
    {
        var currentWriteCount = await _forwardIntegrator.GetCurrentWriteCountAsync();
        var currentTime = Time.time;

        if(_lastPerformanceUpdateTime > 0)
        {
            var deltaWrites = currentWriteCount - _lastForwardWriteCount;
            var deltaTime = currentTime - _lastPerformanceUpdateTime;
            ForwardWritesPerSecond = (long)(deltaWrites / deltaTime);
        }

        _lastForwardWriteCount = currentWriteCount;
        _lastPerformanceUpdateTime = currentTime;
    }
}